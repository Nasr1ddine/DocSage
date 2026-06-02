import os
import time
from pathlib import Path
from typing import Dict, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .ingest import run_ingest
from .pipeline import chat_once
from .retriever import HybridRetriever


load_dotenv(Path(__file__).resolve().parent.parent / ".env")

app = FastAPI(title="DocSage — Internal Docs Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve frontend ────────────────────────────────────────────────────
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    index = _FRONTEND_DIR / "index.html"
    return HTMLResponse(content=index.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> Dict:
    try:
        # lightweight check by constructing a retriever and checking corpus size
        tuned_retriever = HybridRetriever("tuned")
        naive_retriever = HybridRetriever("naive")
        # we approximate docs_indexed via naive collection point count
        from qdrant_client import QdrantClient

        client = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY") or None,
        )
        stats_naive = client.get_collection(os.getenv("QDRANT_COLLECTION_NAIVE", "docsage_naive"))
        stats_tuned = client.get_collection(os.getenv("QDRANT_COLLECTION_TUNED", "docsage_tuned"))
        docs_indexed = (
            stats_naive.vectors_count if hasattr(stats_naive, "vectors_count") else 0
        ) + (stats_tuned.vectors_count if hasattr(stats_tuned, "vectors_count") else 0)
        _ = (tuned_retriever, naive_retriever)
        return {"status": "ok", "qdrant": "connected", "docs_indexed": int(docs_indexed)}
    except Exception:
        return {"status": "degraded", "qdrant": "unreachable", "docs_indexed": 0}


@app.post("/ingest")
def ingest(payload: Dict) -> Dict:
    strategy = payload.get("strategy")
    if strategy not in ("naive", "tuned"):
        raise HTTPException(status_code=400, detail="strategy must be 'naive' or 'tuned'")

    chunks_created, storage_mb, time_seconds = run_ingest(strategy)  # type: ignore[arg-type]
    return {
        "chunks_created": chunks_created,
        "storage_mb": storage_mb,
        "time_seconds": time_seconds,
    }


@app.post("/chat")
def chat(payload: Dict) -> JSONResponse:
    question = payload.get("question")
    collection: Literal["naive", "tuned"] = payload.get("collection", "tuned")
    if not question or collection not in ("naive", "tuned"):
        raise HTTPException(status_code=400, detail="Invalid request.")

    try:
        t0 = time.perf_counter()
        result = chat_once(question, collection)
        t1 = time.perf_counter()

        latency = result["latency"]
        latency["total_ms"] = (t1 - t0) * 1000.0  # ensure total includes API overhead

        response = {
            "answer": result["answer"],
            "citations": result["citations"],
            "latency_ms": latency["total_ms"],
            "retrieval_scores": result["retrieval_scores"],
            "latency_breakdown": latency,
        }
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Chat failed: {type(e).__name__}: {str(e)}",
        )


@app.post("/chat/stream")
def chat_stream(payload: Dict):
    """
    Streaming variant of /chat using Server-Sent Events (SSE).
    Sends incremental chunks of the final answer for a smooth UX.
    """
    question = payload.get("question")
    collection: Literal["naive", "tuned"] = payload.get("collection", "tuned")
    if not question or collection not in ("naive", "tuned"):
        raise HTTPException(status_code=400, detail="Invalid request.")

    try:
        result = chat_once(question, collection)
        answer_text = result["answer"]

        def event_stream():
            # simple character-wise streaming to emulate token streaming
            for ch in answer_text:
                yield f"data: {ch}\n\n"
                time.sleep(0.005)
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Chat failed: {type(e).__name__}: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

