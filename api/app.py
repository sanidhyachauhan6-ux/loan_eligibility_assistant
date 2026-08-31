"""ClaimAssist API — v5 (Session 10). ClaimAssist is complete.

Endpoints:
    GET  /health              -> liveness + active prompt_version
    POST /ask                 -> guarded, traced, grounded, structured answer

THE v4 -> v5 CHANGES (LLMOps + Responsible AI):
  1. PROMPT REGISTRY  — the system prompt is no longer a string in this file;
     it is loaded from prompts/registry.yaml via prompts/loader.py, and the
     active version is selected by env PROMPT_VERSION. Promoting a prompt is
     an .env change — a release, not a code edit.
  2. TRACING          — the openai client is the langfuse.openai DROP-IN
     wrapper, so every LLM generation (latency, tokens, prompt, completion)
     is traced; @observe wraps the /ask handler so the whole request is one
     trace, tagged with prompt_version and the caller's session id.
     If LANGFUSE_* keys are unset, everything degrades to a NO-OP: the app
     runs identically, just untraced.
  3. GUARDRAILS       — middleware order: input guard -> traced LLM call ->
     output guard (guardrails.py). Refusals return
     {answer: <refusal>, refused: true, reason: ...} — never a 500.

Note on streaming: v5's /ask is non-streaming BY DESIGN. The output guard
must see the complete answer before anything leaves the boundary — you cannot
un-stream a leaked phone number. Production systems that stream moderate in
buffered windows; that trade-off is discussed in the deck.
"""
import asyncio
import base64
import re

import httpx
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile, Depends, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import AsyncIterator, Dict, Literal, Optional

from guardrails import REFUSAL, check_input, check_output
from redact import redact
from prompts.loader import load_prompt
import chromadb
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("loanassist.api")

# --- configuration (all via env; see docker-compose.yml / .env) -------------
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:4000/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "local")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-local")
API_KEY = os.getenv("API_KEY", "local-dev-key")
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v1")
CHROMA_DIR = os.environ.get("CHROMA_DIR", "rag/chroma")
RAG_COLLECTION = "eligibility_2"
AUDIT_PATH = os.environ.get("AUDIT_PATH", "/app/logs/audit.jsonl")
TOP_K = 5  # clauses retrieved per question

# ---- resilience knobs (unchanged from v1) -----------------------------------
REQUEST_TIMEOUT = httpx.Timeout(90.0, connect=5.0)
MAX_RETRIES = 2
BACKOFF_BASE_S = 0.5

# --- prompt registry: fail FAST at startup on an unknown version ------------
ACTIVE_PROMPT = load_prompt("answer_grounded", PROMPT_VERSION)

# --- Langfuse tracing: drop-in wrapper, graceful no-op without keys ----------
# With LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST set, the
# langfuse.openai import is a DROP-IN for the openai SDK: same classes, same
# calls, every generation traced. Without keys (or without the package) the
# app must behave identically — observability must never take the product down.
LANGFUSE_ENABLED = bool(
    os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
)
if LANGFUSE_ENABLED:
    try:
        from langfuse.decorators import langfuse_context, observe
        from langfuse.openai import OpenAI  # the drop-in wrapper
        logger.info("Langfuse tracing ENABLED (host=%s)",
                    os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"))
    except Exception as exc:  # missing package, bad keys — degrade, don't die
        logger.warning("Langfuse unavailable (%s) — tracing disabled", exc)
        LANGFUSE_ENABLED = False
if not LANGFUSE_ENABLED:
    from openai import OpenAI  # plain client, no tracing

    langfuse_context = None

    def observe(*_args, **_kwargs):  # no-op decorator, same signature
        def decorator(fn):
            return fn
        return decorator


client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, timeout=60.0,
                max_retries=2)

app = FastAPI(title="ClaimAssist API", version="5.0.0")

# ---- RAG store ----------------------------------------------------------------
# The API process embeds chroma directly (PersistentClient over the directory
# rag/ingest.py wrote). The first query triggers the same ~80 MB ONNX embedding
# model download inside the container — the compose file mounts a volume over
# the cache so it happens once. In production this in-process store becomes a
# vector database SERVICE (pgvector, a managed vector DB) shared by replicas.

_chroma = chromadb.PersistentClient(path=CHROMA_DIR)

def get_collection():
    try:
        return _chroma.get_collection(RAG_COLLECTION)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=503,
            detail=f"RAG store error: {type(exc).__name__}: {exc}",
        )


def retrieve_clauses(question: str) -> list[dict]:
    """Top-K nearest clause chunks for the question.

    Returns [{clause_id, doc, text, distance}] ordered best-first. chroma's
    default space here is L2 over normalised MiniLM embeddings: SMALLER
    distance = closer match. The best (smallest) distance drives `confidence`.
    """
    res = get_collection().query(query_texts=[question], n_results=TOP_K)
    out = []
    for i in range(len(res["ids"][0])):
        out.append(
            {
                "doc": res["metadatas"][0][i]["doc"],
                "section": res["metadatas"][0][i]["section"],
                "rule_id": res["metadatas"][0][i].get("rule_id", "-"),
                "text": res["documents"][0][i],
                "distance": res["distances"][0][i],
            }
        )
    return out

# ---- confidence: retrieval distance -> label ----------------------------------
# Thresholds are CORPUS-SPECIFIC and belong in code review, not folklore.
# For this corpus (short clause chunks, MiniLM, L2): a direct clause hit
# ("licence validity") lands well under 1.0; a vaguely related question sits
# around 1.0-1.4; beyond 1.4 retrieval is guessing. Calibrate against a
# labelled question set before trusting these numbers in production —
# Session 3's promptfoo gate is where such assertions live.
CONF_HIGH_MAX = 1.0    # best distance below this  -> "high"
CONF_MEDIUM_MAX = 1.4  # below this                -> "medium"; else "low"


def confidence_from_distance(best_distance: float) -> str:
    if best_distance < CONF_HIGH_MAX:
        return "high"
    if best_distance < CONF_MEDIUM_MAX:
        return "medium"
    return "low"

# ---- Idempotency-Key cache (unchanged from v1) --------------------------------
IDEMPOTENCY_TTL_S = 600
_idempotency_cache: dict[str, tuple[float, dict]] = {}


def idempotency_get(key: str) -> Optional[dict]:
    entry = _idempotency_cache.get(key)
    if entry is None:
        return None
    stored_at, response = entry
    if time.time() - stored_at > IDEMPOTENCY_TTL_S:
        _idempotency_cache.pop(key, None)
        return None
    return response


def idempotency_put(key: str, response: dict) -> None:
    now = time.time()
    for k in [k for k, (t, _) in _idempotency_cache.items() if now - t > IDEMPOTENCY_TTL_S]:
        _idempotency_cache.pop(k, None)
    _idempotency_cache[key] = (now, response)

# --- models -------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str

class Citation(BaseModel):
    doc: str
    text: str

class AskResponse(BaseModel):
    """v5 contract: refusals are FIRST-CLASS fields, not HTTP errors."""
    answer: str
    decision: Optional[
        Literal[
            "PRE_QUALIFIED",
            "NOT_PRE_QUALIFIED",
            "NEEDS_INFORMATION",
            "MANUAL_REVIEW",
        ]
    ] = None
    citations: list[Citation] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]
    refused: bool = False
    reason: Optional[str] = None
    prompt_version: str = PROMPT_VERSION

class IntakeResult(BaseModel):
    """Structured extraction from an uploaded claim document."""
    document_type: str          # e.g. "garage_estimate", "discharge_summary"
    key_fields: dict            # e.g. {"claim_id": "CLM-1001", "total_inr": 84500}
    source: Literal["vision_model", "local_text_fallback"]
    model: str

# def build_messages(question: str, request_type: str="general") -> tuple[list, list]:

#     system = ACTIVE_PROMPT["text"]

#     if request_type == "rag":
#         rag_results = retrieve_clauses(question)

#         rag_context = [
#             {
#                 "doc": item["doc"],
#                 "text": item["text"],
#             }
#             for item in rag_results
#         ]

#         rag_formatted = "\n\n".join(
#             f"[{item['doc']}]\nRule: {item['text']}"
#             for item in rag_results
#         )

#         rag_prompt = f"""
# {system}

# You are handling a loan eligibility question.

# You MUST return valid JSON only with exactly this structure:

# {{
#     "decision": "PRE_QUALIFIED" | "NOT_PRE_QUALIFIED" |
#                  "NEEDS_INFORMATION" | "MANUAL_REVIEW",
#     "answer": "A clear, user-friendly explanation of the result."
# }}

# Rules:
# - "decision" must be one of the four allowed values.
# - Determine the decision only from the provided eligibility rules
#   and information supplied by the applicant.
# - Do not invent missing applicant information.
# - If required information is missing, use NEEDS_INFORMATION.
# - "answer" is the message shown directly to the applicant.
# - Do not include Markdown.
# - Do not include any fields other than "decision" and "answer".

# Relevant eligibility rules:

# {rag_formatted}
# """

#         messages = [
#             {
#                 "role": "system",
#                 "content": rag_prompt,
#             },
#             {
#                 "role": "user",
#                 "content": question,
#             },
#         ]

#         return messages, rag_context

#     # ---------------- GENERAL CHAT ----------------

#     general_prompt = f"""
# {system}

# This is a general conversational question, not an eligibility
# assessment.

# Answer briefly, factually, and politely.

# Ask the user to stick to loan eligibility related queries only.

# Return valid JSON only:

# {{
#     "answer": "..."
# }}

# Do not include any other fields.
# """

#     messages = [
#         {
#             "role": "system",
#             "content": general_prompt,
#         },
#         {
#             "role": "user",
#             "content": question,
#         },
#     ]

#     return messages, []
def get_history(session_id: str):
    return conversation_store.get(session_id, [])


def save_message(session_id: str, role: str, content: str):
    conversation_store.setdefault(session_id, []).append({
        "role": role,
        "content": content
    })


def build_messages(
    question: str,
    request_type: str,
    session_id: str,
) -> tuple[list, list]:
    history = get_history(session_id)

    system = ACTIVE_PROMPT["text"]

    # ---------------------------------------------------------
    # GENERAL conversation
    # ---------------------------------------------------------
    if request_type == "GENERAL":

        system_prompt = f"""
{system}

This is a LoanAssist conversational interaction.

Do not perform an eligibility assessment unless the user explicitly asks
for one.

Respond naturally and briefly while staying within the LoanAssist role.

For greetings or casual conversation, help guide the customer toward
loan-related assistance.

Return valid JSON only:

{{
    "answer": "..."
}}

Do not include any other fields.
"""

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ]
        messages.extend(history)
        return messages, []

    # ---------------------------------------------------------
    # ELIGIBILITY / RAG
    # ---------------------------------------------------------
    rag_results = retrieve_clauses(question)

    rag_context = [
        {
            "doc": item["doc"],
            "text": item["text"],
            "rule_id": item.get("rule_id", "-"),
        }
        for item in rag_results
    ]

    rag_formatted = "\n\n".join(
        f"[{item.get('rule_id', '-')}] "
        f"[{item['doc']}]\n"
        f"Rule: {item['text']}"
        for item in rag_results
    )

    system_prompt = f"""
{system}

This is an eligibility-related request.

Use ONLY the policy information provided below.

Determine the eligibility decision according to the policy.
Do not invent rules or customer information.

Return valid JSON only:

{{
    "decision": "PRE_QUALIFIED" | "NOT_PRE_QUALIFIED" |
                 "NEEDS_INFORMATION" | "MANUAL_REVIEW",
    "answer": "A clear, user-friendly explanation of the decision."
}}

Rules:

- "decision" MUST contain the eligibility decision.
- "answer" is the message shown directly to the applicant.
- Do not include Markdown.
- Do not include any fields other than "decision" and "answer".
- Apply the policy rules exactly.
- If required information is missing, use NEEDS_INFORMATION.
- Do not assume missing information.

Policy context:

{rag_formatted}
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": question,
        },
    ]
    messages.extend(history)
    return messages, rag_context

CITATION_RE = re.compile(r"\[((?:INCOME|EMP|AGE)-\d+)\]")

def extract_citations(answer: str, retrieved: list[dict]) -> list[Citation]:
    """Turn [M-2.3]-style ids in the answer into verifiable Citation objects.

    Only ids that were actually RETRIEVED (or exist in the store) become
    citations — an id the model invented that matches no chunk is dropped,
    which is itself a faithfulness signal.
    """
    by_id = {c["id"]: c for c in retrieved}
    citations: list[Citation] = []
    for cid in dict.fromkeys(CITATION_RE.findall(answer)):  # unique, ordered
        chunk = by_id.get(cid)
        if chunk is None:
            # cited but not in the top-K: look it up directly in the store so
            # a legitimate citation outside the retrieval window still resolves
            got = get_collection().get(ids=[cid])
            if not got["ids"]:
                continue  # invented id — drop it
            chunk = {
                "id": cid,
                "doc": got["metadatas"][0]["doc"],
                "text": got["documents"][0],
            }
        citations.append(
            Citation(id=cid, doc=chunk["doc"], snippet=chunk["text"][:200])
        )
    return citations

NOT_FOUND_ANSWER = "This is not covered in the policy documents I have access to."

# ---- upstream LLM calls (unchanged mechanics from v1/v2) -----------------------
async def call_llm(messages: list, max_tokens: int = 300, model: str = "") -> str:
    payload = {
        "model": model or LLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {LLM_API_KEY}"}
    last_error: Optional[Exception] = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                r = await client.post(
                    f"{LLM_BASE_URL}/chat/completions", json=payload, headers=headers
                )
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
        except (httpx.TimeoutException, httpx.HTTPError) as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                await asyncio.sleep(BACKOFF_BASE_S * (2 ** attempt))
    raise HTTPException(
        status_code=503,
        detail=f"LLM upstream unavailable after {MAX_RETRIES + 1} attempts: "
               f"{type(last_error).__name__}",
    )

def audit_log(entry: dict):
    """Append one audit event as a JSON line."""
    os.makedirs(os.path.dirname(AUDIT_PATH), exist_ok=True)

    with open(AUDIT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# --- endpoints -------------------------------------------------------------------
@app.get("/health")
async def health():
    try:
        rag_chunks = get_collection().count()
    except HTTPException:
        rag_chunks = 0
    return {
        "status": "ok",
        "llm_base_url": LLM_BASE_URL,
        "model": LLM_MODEL,
        "prompt_version": PROMPT_VERSION,
        "langfuse_enabled": LANGFUSE_ENABLED,
        "rag_chunks": rag_chunks
    }

@app.post("/ask", response_model=AskResponse)
@observe(name="loanassist-ask")  # one trace per /ask request
async def ask(
    req: AskRequest,
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
    x_session_id: Optional[str] = Header(default=None, alias="X-Session-Id")
):
    """Middleware order: input guard -> traced LLM call -> output guard."""
    session_id = x_session_id or f"anon-{uuid.uuid4().hex[:8]}"

    # Tag the trace so it is filterable in Langfuse: prompt_version drives the
    # A/B comparison; session_id groups a conversation end-to-end.
    if LANGFUSE_ENABLED:
        langfuse_context.update_current_trace(
            session_id=session_id,
            tags=[f"prompt_version:{PROMPT_VERSION}", "app:loanassist"],
        )
    # ---- layer 1: input guard (before the model sees anything) --------------
    t0 = time.perf_counter()
    verdict = check_input(req.question, client, LLM_MODEL)
    print(verdict)
    if not verdict["allowed"]:
        latency_ms = round((time.perf_counter() - t0) * 1000)
        audit_log({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "session_id": session_id,
            "requested_model": LLM_MODEL,
            "actual_model": None,
            "prompt_redacted": redact(req.question),
            "response_redacted": verdict["reason"],
            "latency_ms": latency_ms,
            "max_tokens": None,
            "prompt_version": PROMPT_VERSION
        })
        return AskResponse(
            answer=REFUSAL,
            citations=[],
            confidence="high",
            refused=True,
            reason=verdict["reason"],
            prompt_version=PROMPT_VERSION,
        )
    elif verdict["allowed"] and verdict["type"]=="GENERAL":
        t0 = time.perf_counter()
        try:
            # messages, rag_context = build_messages(req.question, request_type="general")
            messages, rag_context = build_messages(req.question, request_type=verdict["type"], session_id=session_id)
            completion = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                max_tokens=300,
            )
            content = (completion.choices[0].message.content or "").strip()
            answer = content
        except Exception as exc:
            latency_ms = round((time.perf_counter() - t0) * 1000)
            audit_log({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "session_id": session_id,
                "requested_model": LLM_MODEL,
                "actual_model": (
                    completion.model
                    if completion is not None
                    else "failed_at_llm_call"
                ),
                "prompt_redacted": redact(req.question),
                "response_redacted": {exc},
                "latency_ms": latency_ms,
                "prompt_version": PROMPT_VERSION
            })
            raise HTTPException(status_code=502, detail=f"Upstream LLM error: {exc}")
        out = check_output(answer)
        if not out["text"]:
            latency_ms = round((time.perf_counter() - t0) * 1000)
            audit_log({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "session_id": session_id,
                "requested_model": LLM_MODEL,
                "actual_model": (
                    completion.model
                    if completion is not None
                    else "failed_at_llm_call"
                ),
                "prompt_redacted": redact(req.question),
                "response_redacted": "",
                "citations": rag_context,
                "latency_ms": latency_ms,
                "prompt_version": PROMPT_VERSION
            })
            return AskResponse(
                answer="I could not generate an answer. Please contact the helpline.",
                citations=[],
                confidence="low",
                refused=False,
                reason="empty_model_output",
                prompt_version=PROMPT_VERSION,
            )
        latency_ms = round((time.perf_counter() - t0) * 1000)
        audit_log({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "session_id": session_id,
            "requested_model": LLM_MODEL,
            "actual_model": (
                completion.model
                if completion is not None
                else "failed_at_llm_call"
            ),
            "prompt_redacted": redact(req.question),
            "response_redacted": redact(out["text"]),
            "latency_ms": latency_ms,
            "prompt_version": PROMPT_VERSION
        })
        confidence = "medium"
        return AskResponse(
            answer=answer,
            citations=[],
            confidence="high" if out["refused"] else confidence,
            refused=out["refused"],
            reason=out["reason"],
            prompt_version=PROMPT_VERSION

        )
    else:
        """RAG-grounded, cited, structured answer — the v3 core.

        Pipeline: retrieve top-4 clauses -> grounded prompt (with NOT_IN_POLICY
        escape) -> LLM via the LiteLLM proxy -> post-process: citations extracted
        by regex, confidence from best retrieval distance, honest not-found state.
        """
        if idempotency_key:
            cached = idempotency_get(idempotency_key)
            if cached is not None:
                return AskResponse(**cached)
        
        # raw = await call_llm(build_messages(req.question))

        # ---- layer 2: the traced LLM call ---------------------------------------
        # The langfuse.openai wrapper records this generation (model, latency,
        # token usage, prompt, completion) inside the current trace automatically.
        t0 = time.perf_counter()
        try:
            # messages, rag_context = build_messages(req.question, request_type="rag")
            messages, rag_context = build_messages(req.question, request_type=verdict["type"], session_id=session_id)
            completion = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                max_tokens=3000,
            )
            content = (completion.choices[0].message.content or "").strip()
            output = json.loads(content)
            decision = output["decision"]
            answer = output["answer"]
        except Exception as exc:
            latency_ms = round((time.perf_counter() - t0) * 1000)
            audit_log({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "session_id": session_id,
                "requested_model": LLM_MODEL,
                "actual_model": (
                    completion.model
                    if completion is not None
                    else "failed_at_llm_call"
                ),
                "prompt_redacted": redact(req.question),
                "response_redacted": {exc},
                "latency_ms": latency_ms,
                "prompt_version": PROMPT_VERSION
            })
            raise HTTPException(status_code=502, detail=f"Upstream LLM error: {exc}")
        # logger.info("llm_call ok latency_ms=%s prompt_version=%s session=%s",
        #             latency_ms, PROMPT_VERSION, session_id)

        # ---- layer 3: output guard (before anything leaves the boundary) --------
        out = check_output(answer)
        if not out["text"]:
            latency_ms = round((time.perf_counter() - t0) * 1000)
            audit_log({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "session_id": session_id,
                "requested_model": LLM_MODEL,
                "actual_model": (
                    completion.model
                    if completion is not None
                    else "failed_at_llm_call"
                ),
                "prompt_redacted": redact(req.question),
                "response_redacted": "",
                "citations": rag_context,
                "latency_ms": latency_ms,
                "prompt_version": PROMPT_VERSION
            })
            return AskResponse(
                answer="I could not generate an answer. Please contact the helpline.",
                decision="NEEDS_INFORMATION",
                citations=[],
                confidence="low",
                refused=False,
                reason="empty_model_output",
                prompt_version=PROMPT_VERSION,
            )
        latency_ms = round((time.perf_counter() - t0) * 1000)
        audit_log({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "session_id": session_id,
            "requested_model": LLM_MODEL,
            "actual_model": (
                completion.model
                if completion is not None
                else "failed_at_llm_call"
            ),
            "prompt_redacted": redact(req.question),
            "response_redacted": redact(out["text"]),
            "citations": rag_context,
            "latency_ms": latency_ms,
            "prompt_version": PROMPT_VERSION
        })
        if idempotency_key:
            idempotency_put(idempotency_key, final.model_dump())

        confidence = "medium"
        return AskResponse(
            answer=out["text"],
            decision=decision,
            citations=rag_context,
            confidence="high" if out["refused"] else confidence,
            refused=out["refused"],
            reason=out["reason"],
            prompt_version=PROMPT_VERSION
        )

REQUESTS = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["endpoint", "status"],
)

LATENCY = Histogram(
    "llm_request_latency_seconds",
    "LLM request latency in seconds",
    ["endpoint"],
    buckets=(0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32),
)

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Times requests and records metrics for /ask and /health."""

    t0 = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        # Don't record latency here unless you also want
        # unhandled 5xx requests represented in the histogram.
        raise

    dt = time.perf_counter() - t0
    path = request.url.path

    # Keep metric label cardinality bounded.
    if path in ("/ask", "/health"):
        REQUESTS.labels(
            endpoint=path,
            status=str(response.status_code),
        ).inc()

    # Only measure actual LLM request latency.
    if path == "/ask":
        LATENCY.labels(
            endpoint=path,
        ).observe(dt)

    return response

@app.get("/metrics")
def metrics():
    """Prometheus scrapes this endpoint. Plain text, one metric per line."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
