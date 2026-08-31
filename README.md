# Session 8 Lab — ClaimAssist v3: RAG with Citations + Multimodal Intake

**Track 2.B · Session 3 (Day 8) — Multimodal Input & RAG-Backed Features**

## Goal

Upgrade **ClaimAssist v2 → v3**. The claim record told the model *what
happened*; today the policy documents tell it *why* — with verifiable
citations:

- **`rag/ingest.py`** chunks `data/policies/*.md` **by clause** and stores
  them in a persistent **chromadb** index (`rag/chroma/`);
- **`POST /ask`** now retrieves the top-4 clauses, grounds the LLM in that
  numbered context, and returns
  `{answer, citations: [{clause_id, doc, snippet}], confidence, not_found}` —
  including an **honest NOT_IN_POLICY state** for out-of-corpus questions;
- **`POST /intake`** extracts structured fields from an uploaded claim
  document image — a **vision model via the proxy** (cloud path) or the
  **text-sidecar fallback** (local path, the 0.5B model is text-only);
- the **Streamlit UI** renders citation chips, a coloured confidence badge,
  the not-in-policy state, and an Upload document tab.

Time budget: **~60 minutes** (Step 0: 8 · Step 1: 8 · Step 2: 12 ·
Step 3: 8 · Step 4: 8 · Step 5: 12 · Step 6: 4).

## Architecture

```
        streamlit run ui/chat_app.py                        docker compose
 ┌─────────────────────┐        ┌──────────────────────────┐        ┌───────────────┐
 │  Streamlit  :8501   │  HTTP  │   ClaimAssist API :8000  │ OpenAI │ litellm :4000 │
 │  chat + citations   ├───────►│  POST /ask   (RAG+cite)  ├───────►│  the proxy    │
 │  confidence badge   │◄───────┤  POST /intake (upload)   │◄───────┤  (v2, as-is)  │
 │  upload tab         │        │  GET /claims · /ask/stream        └───────┬───────┘
 └─────────────────────┘        └───────┬──────────┬───────┘          local │ cloud
                                        │ embedded │ reads          ┌───────▼───────┐
                             rag/chroma/ (chromadb) │ data/         │  model :8090  │
                             written by rag/ingest.py               │  Qwen 0.5B    │
                                                                    └───────────────┘
```

chroma is **embedded in the API process** — no extra container. The index
directory `./rag/chroma` is written by the ingest script on the host and
bind-mounted into the api container.

---

## Step 0 — Setup (8 min)

```bash
cd session8_lab

cp .env.example .env
# Windows PowerShell: Copy-Item .env.example .env

# host venv: ingest script + UI + sample generator
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install chromadb==0.5.23 streamlit requests pillow

# generate the two sample intake documents (PNG + .txt sidecar each)
python scripts/make_samples.py
ls data/samples/
```

> **First-run download:** chromadb's default embedding function pulls a small
> ONNX embedding model (**~80 MB**, all-MiniLM-L6-v2) to
> `~/.cache/chroma/onnx_models` the first time you embed or query. It happens
> once on the host (Step 1) and once in the api container (first question —
> the compose file persists that cache in a volume).

---

## Step 1 — Ingest: chunk the policies BY CLAUSE (8 min)

```bash
python rag/ingest.py
```

Expected output: per-file clause counts (15 each), then
`Stored 45 clause chunks in rag/chroma/` and five sample chunks.

**Inspect what just happened** — open `rag/ingest.py` and find:

- the clause regex (`^[MHP]-\d+\.\d+ …:`) — the chunk boundary is the
  **clause heading**, not a character count;
- each chunk's **id IS the clause id** (`M-2.3`) and its metadata carries
  `{doc, clause_id}` — this is exactly what makes `[M-2.3]` in an answer
  resolvable to one quotable chunk. Chunking strategy *is* citation strategy;
- delete + recreate on every run: re-ingesting after editing a policy never
  leaves stale chunks behind.

Re-run the script to confirm idempotency (same count, no duplicates).

---

## Step 2 — Start the stack and get a cited answer (12 min)

```bash
docker compose up -d --build
docker compose ps        # wait until model, litellm AND api are (healthy)
# macOS: no `watch` by default — `brew install watch`, or re-run compose ps

curl -s localhost:8000/health | python3 -m json.tool
# → "rag_chunks": 45  — the api sees the index you built in Step 1
```

Now the question this whole session exists for:

```bash
curl -s localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Why was claim CLM-1003 rejected?","claim_id":"CLM-1003"}' \
  | python3 -m json.tool
```

**Checkpoint — the answer must cite `[M-2.3]`** (licence validity). Inspect
the JSON:

- `citations` — `[{clause_id: "M-2.3", doc: "motor_policy.md", snippet: …}]`:
  the first 200 chars of the actual clause, attached as evidence;
- `confidence` — derived from the **best retrieval distance**, not from the
  model's tone;
- the first question is slower: the api container downloads its ONNX
  embedding model (see Step 0 note), then queries are instant.

The 0.5B model occasionally words things clumsily — but the citation ids are
extracted by regex and verified against the store, so a cited clause is
always a real clause.

---

## Step 3 — The honesty test: NOT_IN_POLICY (8 min)

Ask something the corpus genuinely does not cover:

```bash
curl -s localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Does my policy cover pet insurance?"}' \
  | python3 -m json.tool
```

Expected:

```json
{
  "answer": "This is not covered in the policy documents I have access to.",
  "citations": [],
  "confidence": "low",
  "not_found": true
}
```

**Discuss (2 min, pairs):** why does refusing beat hallucinating here? An
invented "pet insurance clause P-9.4" is not a wrong answer, it is a
**liability** — a customer acts on it, an auditor asks where it came from,
and no such clause exists. The `NOT_IN_POLICY` escape hatch in the prompt
(see `GROUNDED_SYSTEM` in `api/app.py`) plus the structured `not_found` state
turn "I don't know" into a designed product behaviour. Note it is the small
model following the instruction — test this boundary in your evals (Step 6).

---

## Step 4 — Confidence: watch it drop (8 min)

```bash
# precise question → the best clause is a near-direct hit
curl -s localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question":"What does the policy say about driving licence validity?"}' \
  | python3 -m json.tool | grep -E 'confidence'

# vague question → retrieval still returns SOMETHING, but further away
curl -s localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question":"What generally happens with my stuff and the paperwork?"}' \
  | python3 -m json.tool | grep -E 'confidence'
```

**Where the thresholds live:** `api/app.py`, constants `CONF_HIGH_MAX = 1.0`
and `CONF_MEDIUM_MAX = 1.4`, with a comment block explaining that they are
corpus-specific (L2 distance over MiniLM embeddings, short clause chunks) and
must be **calibrated against a labelled question set** before anyone trusts
them. Try editing a threshold and re-running (`docker compose up -d --build
api`) to see the label flip — that is exactly why calibration matters.

---

## Step 5 — The UI: citation chips, badge, upload tab (12 min)

```bash
source .venv/bin/activate
streamlit run ui/chat_app.py     # http://localhost:8501
```

**Chat tab:**

- select **CLM-1003** → "Why was my claim rejected?" — answer, a coloured
  confidence badge, and **citation chips**: expanders labelled
  `[M-2.3] — motor_policy.md` with the clause text inside;
- ask "Does my policy cover pet insurance?" — the honest not-in-policy
  state, rendered deliberately (info box), not as an error.

**Upload document tab:**

- upload `data/samples/garage_estimate_clm1001.png` → Extract fields;
- with `OPENROUTER_API_KEY` set, the UI shows *"Extracted by vision model:
  openrouter-mini"* — the API base64-encodes the image (no sidecar);
- offline: clear `VISION_MODEL` and set `LLM_MODEL=qwen-local` to use the
  matching `.txt` sidecar instead;
- inspect the extracted `{document_type, key_fields}` JSON — this is the
  Pydantic `IntakeResult` contract; try the discharge summary too.

> ### Vision intake (OpenRouter by default)
>
> With `OPENROUTER_API_KEY` in `.env`, `/intake` reads the uploaded PNG via
> `openrouter-mini` (no sidecar). Upload a sample image and the UI banner
> should say *"Extracted by vision model: openrouter-mini"*.
>
> Offline sidecar path: set `VISION_MODEL=` (empty) and `LLM_MODEL=qwen-local`
> in `.env`, then `docker compose up -d --force-recreate api`.

---

## Step 6 — Eval tie-back + cleanup (4 min)

Session 3 built a promptfoo gate in CI. RAG adds a new failure class —
**unfaithful answers** — so the gate gains a faithfulness assertion: a
question with a *known* clause must cite it, and out-of-corpus questions must
refuse. Add to your Session 3 `promptfooconfig.yaml`:

```yaml
tests:
  - description: "faithfulness: rejection answer must cite M-2.3"
    vars:
      question: "Why was claim CLM-1003 rejected?"
    assert:
      - type: contains          # citation present in the answer
        value: "[M-2.3]"
      - type: javascript        # citation resolved by the API post-processor
        value: JSON.parse(context.response).citations.some(c => c.clause_id === "M-2.3")
  - description: "honesty: out-of-corpus question must refuse"
    vars:
      question: "Does my policy cover pet insurance?"
    assert:
      - type: javascript
        value: JSON.parse(context.response).not_found === true
```

Every prompt or chunking change now has to keep citations and honesty intact
to reach main — evaluation as a merge gate, extended to RAG.

```bash
docker compose down        # keep hf_cache + chroma_cache volumes for S9
```

Keep the directory — **Session 9 reuses this chroma store** as the
`search_policy` MCP tool.

---

## Local → production mapping

| In this lab | In production |
|---|---|
| Embedded chromadb (`PersistentClient` on `rag/chroma/`) | A vector database service: pgvector, or a managed vector DB (Pinecone, Weaviate, Vertex/Bedrock vector stores) shared by replicas |
| Chunking by clause (regex on `M-x.y` headings) | Domain-aware chunking strategies: by section/clause/table for contracts & policies, semantic splitting for prose — chunking is a product decision |
| Default ONNX embedding model (~80 MB, local) | A pinned embedding-model version served at scale — and re-embedding as a managed migration when it changes |
| `CONF_HIGH_MAX` / `CONF_MEDIUM_MAX` constants | Calibrated confidence scoring against labelled sets, monitored for drift |
| `NOT_IN_POLICY` escape + `not_found` state | Refusal/grounding policies enforced by evals and guardrails (Session 10) |
| `/intake` with a sidecar fallback | Document-AI pipelines: OCR/vision extraction, schema validation, human review queues for low-confidence extractions |
| Faithfulness asserts in promptfoo | RAG eval suites (faithfulness, answer relevance, retrieval recall) gating every prompt/chunking change |

## Deliverables

1. **Cited answer** — screenshot (UI or curl JSON) of the CLM-1003 answer
   citing `[M-2.3]` with its citations array (Step 2/5).
2. **NOT_IN_POLICY proof** — the pet-insurance response JSON with
   `not_found: true` (Step 3).
3. **Intake extraction** — the `IntakeResult` JSON for one sample document,
   either path (`local_text_fallback` or `vision_model`) (Step 5).
4. **Written answer** — one paragraph: what does chunking **by clause** buy
   over fixed-size chunks for policy documents — and what would you lose if a
   single clause were longer than your context budget?

Keep this directory — Session 9 adds MCP tool-use on this same stack.
