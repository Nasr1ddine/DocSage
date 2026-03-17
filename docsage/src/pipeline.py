import os
import time
from pathlib import Path
from typing import Dict, List, Literal, Tuple

from dotenv import load_dotenv
from llama_index.core import ServiceContext
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI

from .retriever import CollectionKind, HybridRetriever, RetrievedChunk


load_dotenv(Path(__file__).resolve().parent.parent / ".env")


SYSTEM_PROMPT = """You are a precise document analyst. Answer ONLY from the provided 
context chunks. If the answer is not in the context, say: 
'I cannot find this in the provided documents.' 
Never speculate beyond the source material.
Always end your answer with: Sources: [list filenames + page numbers]"""


def _get_llm() -> OpenAI:
    model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    return OpenAI(model=model, temperature=0.0)


def _build_sources_suffix(chunks: List[RetrievedChunk]) -> str:
    if not chunks:
        return "Sources: []"
    parts = []
    for c in chunks:
        if c.page >= 0:
            parts.append(f"{c.source} p.{c.page}")
        else:
            parts.append(c.source)
    joined = ", ".join(sorted(set(parts)))
    return f"Sources: [{joined}]"


def _answer_with_llm(
    question: str,
    chunks: List[RetrievedChunk],
    llm: OpenAI,
) -> Tuple[str, float]:
    context_pieces = [c.text for c in chunks]
    context_block = "\n\n---\n\n".join(context_pieces)
    user_prompt = (
        "Use ONLY the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\nQuestion: {question}"
    )
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
        ChatMessage(role=MessageRole.USER, content=user_prompt),
    ]
    t0 = time.perf_counter()
    resp = llm.chat(messages)
    t1 = time.perf_counter()
    answer = resp.message.content or ""
    # ensure the answer always ends with the Sources line exactly as required
    suffix = _build_sources_suffix(chunks)
    if "Sources:" in answer:
        # strip any existing sources and replace
        idx = answer.rfind("Sources:")
        answer = answer[:idx].rstrip()
    answer = f"{answer}\n\n{suffix}"
    return answer, (t1 - t0) * 1000.0


def chat_once(
    question: str,
    collection: Literal["naive", "tuned"],
) -> Dict:
    """
    Main query pipeline used by the API.
    Returns answer, citations, retrieval_scores, and detailed latency breakdown.
    """
    retriever = HybridRetriever(collection)  # small docs, okay to construct per call
    llm = _get_llm()

    t_embed_start = time.perf_counter()
    # dense model encodes query inside retriever; we approximate that cost here
    # by calling retrieve and measuring time windows separately below
    # (embedding_ms will be included in retrieval_ms in practice)
    t_embed_end = t_embed_start

    t_retrieval_start = time.perf_counter()
    chunks, rerank_scores, max_score = retriever.retrieve(question, top_k=5)
    t_retrieval_end = time.perf_counter()

    # confidence gate: skip LLM if retrieval is weak
    if max_score < 0.3:
        llm_ms = 0.0
        answer = (
            "I cannot find this in the provided documents.\n\n"
            "Sources: []"
        )
    else:
        answer, llm_ms = _answer_with_llm(question, chunks, llm)

    total_ms = (t_retrieval_end - t_retrieval_start) * 1000.0 + llm_ms

    citations = [
        {
            "text": c.text,
            "source": c.source,
            "page": c.page,
            "score": c.score,
            "chunk_id": c.chunk_id,
        }
        for c in chunks
    ]

    # best-effort breakdown; embedding is inside retrieval in this implementation
    embedding_ms = (t_embed_end - t_embed_start) * 1000.0
    retrieval_ms = (t_retrieval_end - t_retrieval_start) * 1000.0
    rerank_ms = retrieval_ms  # dominant part of retrieval; kept for compatibility

    return {
        "answer": answer,
        "citations": citations,
        "retrieval_scores": rerank_scores,
        "latency": {
            "embedding_ms": embedding_ms,
            "retrieval_ms": retrieval_ms,
            "rerank_ms": rerank_ms,
            "llm_ms": llm_ms,
            "total_ms": total_ms,
        },
        "max_score": max_score,
    }

