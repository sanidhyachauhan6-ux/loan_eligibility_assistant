# app.py — Session 6 model server: the OpenAI-compatible shim (llm-server:openai)
#
# The same Qwen 0.5B container from Track 2.A, now speaking the OpenAI protocol:
#   GET  /health                → {"status": "ok"}
#   POST /v1/chat/completions   → OpenAI-shaped JSON, or SSE stream when stream=true
#
# Because it implements the OpenAI wire format, this local container is a
# drop-in target for the openai SDK, LiteLLM (Session 7) and the Vercel AI SDK
# (Session 7) — the "OpenAI protocol as lingua franca" thread from Session 1.
import json
import os
import time
import uuid

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, pipeline

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
SERVED_MODEL = os.environ.get("SERVED_MODEL", "qwen-local")

app = FastAPI(title="llm-server:openai — OpenAI-compatible local model server")

# Load ONCE at startup (Session 1 discipline). The tokenizer is kept as a
# separate handle so usage token counts can be computed exactly, the way a
# real provider reports them.
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
generator = pipeline("text-generation", model=MODEL_ID, tokenizer=tokenizer)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = SERVED_MODEL
    messages: list[ChatMessage]
    max_tokens: int = Field(default=256, ge=1, le=1024)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False


@app.get("/health")
def health():
    """Probe target: answers only once the model is loaded and ready."""
    return {"status": "ok"}


def _generate(req: ChatCompletionRequest):
    """Run the pipeline once; return (text, prompt_tokens, completion_tokens)."""
    prompt = tokenizer.apply_chat_template(
        [m.model_dump() for m in req.messages],
        tokenize=False,
        add_generation_prompt=True,
    )
    out = generator(
        prompt,
        max_new_tokens=req.max_tokens,
        do_sample=req.temperature > 0,
        temperature=max(req.temperature, 1e-3),
        return_full_text=False,
    )
    text = out[0]["generated_text"].strip()
    # usage is counted with the real tokenizer — exactly what providers bill on
    prompt_tokens = len(tokenizer.encode(prompt))
    completion_tokens = len(tokenizer.encode(text)) if text else 0
    return text, prompt_tokens, completion_tokens


def _sse_stream(text: str, comp_id: str, created: int, model: str):
    """Stream the (already generated) text out as OpenAI chat.completion.chunk
    SSE events, word by word.

    NOTE — teaching simplification: we generate the FULL text first, then
    stream it out in chunks. On CPU with a 0.5B model this is simple and
    reliable. A production inference server (vLLM, Session 1) streams true
    token-by-token WHILE the model generates, which is where the
    time-to-first-token advantage really comes from. The wire format below is
    identical either way — that is the point of the protocol.
    """
    def chunk(delta: dict, finish_reason=None) -> str:
        payload = {
            "id": comp_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        # SSE framing: a "data: " prefix and a blank line terminate each event
        return "data: " + json.dumps(payload) + "\n\n"

    # First chunk carries the assistant role, exactly as OpenAI's API does.
    yield chunk({"role": "assistant", "content": ""})
    words = text.split(" ")
    for i, word in enumerate(words):
        piece = word + (" " if i < len(words) - 1 else "")
        yield chunk({"content": piece})
        time.sleep(0.02)  # pacing so streaming is visible in curl / the UI
    yield chunk({}, finish_reason="stop")
    yield "data: [DONE]\n\n"  # the OpenAI stream-termination sentinel


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    text, prompt_tokens, completion_tokens = _generate(req)
    comp_id = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = int(time.time())

    if req.stream:
        return StreamingResponse(
            _sse_stream(text, comp_id, created, req.model),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",     # SSE must never be cached
                "X-Accel-Buffering": "no",       # tell reverse proxies not to buffer
            },
        )

    # Non-streaming: the classic OpenAI chat.completion response shape.
    return {
        "id": comp_id,
        "object": "chat.completion",
        "created": created,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
