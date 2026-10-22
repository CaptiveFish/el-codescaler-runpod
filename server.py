"""CodeScaler-4B scoring worker for a Runpod load-balancing serverless endpoint.

Serves exactly the contract consumed by crates/el-core/src/adapters/selector/mod.rs:

  POST /score  {"request_id", "messages":[{"role","content"}], "candidates":
               [{"candidate_id","generation_index","content"}], "model"?}
  -> 200 {"scores":{candidate_id: float in [0,1]}, "ranking":[ids best first],
          "raw_scores":{candidate_id: logit}, "model", "version", "latency_ms"}
  GET  /ping   -> 200 {"status":"healthy",...} when the model is loaded, 204 while it is
                  still loading, 503 if loading failed (Runpod LB rule)

Scores are sigmoid(raw reward logit): the Bradley-Terry probability scale of the
reward model. Ranking order is score descending, then smallest generation_index.
Authentication is enforced by the Runpod edge (Bearer RUNPOD_API_KEY); this process
never sees or logs credentials. Served by FastAPI/uvicorn; the model loads in a background
thread so /ping answers at once. torch/transformers are imported lazily by the real scorer.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Callable, Sequence

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

MODEL_ID = os.environ.get("CODESCALER_MODEL", "LARK-Lab/CodeScaler-4B")
MAX_LENGTH = int(os.environ.get("CODESCALER_MAX_LENGTH", "4096"))
RUNPOD_LB_MAX_PAYLOAD_BYTES = 30 * 1024 * 1024

Scorer = Callable[[Sequence[list[dict]]], list[float]]


class RequestError(ValueError):
    pass


def message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    if content is None:
        return ""
    raise RequestError("message content must be a string or a list of text parts")


def parse_request(body: object) -> tuple[list[dict], list[dict]]:
    if not isinstance(body, dict):
        raise RequestError("request body must be a JSON object")
    messages = body.get("messages")
    candidates = body.get("candidates")
    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a non-empty array")
    if not isinstance(candidates, list) or not candidates:
        raise RequestError("candidates must be a non-empty array")
    conv = []
    for m in messages:
        if not isinstance(m, dict) or not isinstance(m.get("role"), str):
            raise RequestError("each message needs a string role")
        conv.append({"role": m["role"], "content": message_text(m.get("content"))})
    seen = set()
    cands = []
    for c in candidates:
        if not isinstance(c, dict):
            raise RequestError("each candidate must be an object")
        cid, idx, content = c.get("candidate_id"), c.get("generation_index"), c.get("content")
        if not isinstance(cid, str) or not cid:
            raise RequestError("candidate_id must be a non-empty string")
        if cid in seen:
            raise RequestError(f"duplicate candidate_id {cid!r}")
        if not isinstance(idx, int) or isinstance(idx, bool):
            raise RequestError(f"generation_index for {cid!r} must be an integer")
        if not isinstance(content, str):
            raise RequestError(f"content for {cid!r} must be a string")
        seen.add(cid)
        cands.append({"candidate_id": cid, "generation_index": idx, "content": content})
    return conv, cands


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def score_request(body: object, scorer: Scorer, model: str, version: str) -> dict:
    started = time.monotonic()
    conv, cands = parse_request(body)
    convs = [conv + [{"role": "assistant", "content": c["content"]}] for c in cands]
    raw = scorer(convs)
    if len(raw) != len(cands) or not all(isinstance(r, float) and math.isfinite(r) for r in raw):
        raise RuntimeError("scorer returned an invalid score vector")
    scores = {c["candidate_id"]: sigmoid(r) for c, r in zip(cands, raw)}
    order = sorted(cands, key=lambda c: (-scores[c["candidate_id"]], c["generation_index"]))
    return {
        "scores": scores,
        "ranking": [c["candidate_id"] for c in order],
        "raw_scores": {c["candidate_id"]: r for c, r in zip(cands, raw)},
        "model": model,
        "version": version,
        "latency_ms": int((time.monotonic() - started) * 1000),
    }


class TransformersScorer:

    def __init__(self, model_id: str = MODEL_ID, max_length: int = MAX_LENGTH):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        revision = os.environ.get("CODESCALER_REVISION") or None
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.tokenizer.truncation_side = "left"  # keep the candidate at the end
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            num_labels=1,
        ).to(self.device)
        self.model.eval()
        self.max_length = max_length
        self.version = getattr(self.model.config, "_commit_hash", None) or revision or "unknown"
        self.lock = threading.Lock()

    def __call__(self, convs: Sequence[list[dict]]) -> list[float]:
        # One forward pass per candidate, as in the model card: no padding, so the
        # reward head always reads the real last token whatever pad_token_id is.
        scores = []
        with self.lock, self._torch.no_grad():
            for conv in convs:
                text = self.tokenizer.apply_chat_template(conv, tokenize=False)
                toks = self.tokenizer(
                    text, truncation=True, max_length=self.max_length, return_tensors="pt"
                ).to(self.device)
                scores.append(float(self.model(**toks).logits[0][0].float().cpu()))
        return scores


class State:
    def __init__(self):
        self.scorer: Scorer | None = None
        self.version = "unloaded"
        self.load_error: str | None = None

def load_model(state: State, factory: Callable[[], object] | None = None) -> None:
    """Load the scorer into ``state``; never raises (a failure is recorded for /ping)."""
    try:
        scorer = (factory or TransformersScorer)()
        state.version, state.scorer = scorer.version, scorer
        print(f"loaded {MODEL_ID} version={state.version} device={scorer.device}", flush=True)
    except Exception as e:
        state.load_error = f"{type(e).__name__}: {e}"
        print(f"model load failed: {state.load_error}", flush=True)

def _json(code: int, payload: dict | None = None) -> Response:
    # json.dumps default separators keep response bytes identical to the previous
    # stdlib server (Starlette's JSONResponse would compact them).
    if payload is None:
        return Response(status_code=code)
    return Response(content=json.dumps(payload).encode(), status_code=code, media_type="application/json")

def create_app(
    state: State | None = None,
    model: str = MODEL_ID,
    loader: Callable[[State], None] = load_model,
    start_loader: bool = True,
) -> FastAPI:
    state = state if state is not None else State()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if start_loader:
            thread = threading.Thread(target=loader, args=(state,), daemon=True)
            app.state.loader_thread = thread
            thread.start()
        yield

    app = FastAPI(
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        redirect_slashes=False,
    )
    app.state.model_state = state

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> Response:
        # Unknown path or wrong method on a known path: the old server answered 404 for both.
        if exc.status_code in (404, 405):
            return _json(404, {"error": "not found"})
        return _json(exc.status_code, {"error": str(exc.detail)})

    @app.get("/ping")
    async def ping() -> Response:
        if state.load_error:
            return _json(503, {"status": "unhealthy", "error": state.load_error})
        if state.scorer is None:
            return _json(204)
        return _json(200, {"status": "healthy", "model": model, "version": state.version})

    @app.post("/score")
    async def score(request: Request) -> Response:
        if state.scorer is None:
            return _json(503, {"error": "model not loaded"})
        try:
            length = int(request.headers.get("content-length") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > RUNPOD_LB_MAX_PAYLOAD_BYTES:
            return _json(400, {"error": "missing or oversized body"})
        try:
            body = json.loads(await request.body())
            result = await run_in_threadpool(score_request, body, state.scorer, model, state.version)
            return _json(200, result)
        except (RequestError, json.JSONDecodeError) as e:
            return _json(400, {"error": str(e)})
        except Exception as e:
            return _json(500, {"error": f"{type(e).__name__}: {e}"})

    return app

app = create_app()

def main():
    port = int(os.environ.get("PORT", "8000"))
    print(f"listening on :{port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
