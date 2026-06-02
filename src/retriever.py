import os
import time
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

from pathlib import Path

from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


load_dotenv(Path(__file__).resolve().parent.parent / ".env")


CollectionKind = Literal["naive", "tuned"]


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int
    score: float
    chunk_id: str


class HybridRetriever:
    """
    Dense + BM25 + Cross-Encoder reranking over a Qdrant collection.

    WARNING: dense-only retrieval tends to fail on exact-match queries
    (ticker symbols, dates, specific numbers). BM25 is critical to
    catch those lexical matches and is fused here via RRF.
    """

    def __init__(self, kind: CollectionKind):
        self.kind = kind
        self.client = self._init_qdrant()
        self.collection_name = self._get_collection_name(kind)

        # embeddings: use the same OpenAI model used during ingestion
        model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.dense_model = OpenAIEmbedding(model=model_name)
        # cross-encoder reranker
        self.reranker = CrossEncoder(
            os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        )

        # BM25 corpus (lazy init)
        self._bm25 = None
        self._bm25_tokens: List[List[str]] = []
        self._bm25_ids: List[str] = []

    @staticmethod
    def _init_qdrant() -> QdrantClient:
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY") or None
        return QdrantClient(url=url, api_key=api_key)

    @staticmethod
    def _get_collection_name(kind: CollectionKind) -> str:
        if kind == "naive":
            return os.getenv("QDRANT_COLLECTION_NAIVE", "docsage_naive")
        return os.getenv("QDRANT_COLLECTION_TUNED", "docsage_tuned")

    def _ensure_bm25(self) -> None:
        if self._bm25 is not None:
            return
        texts: List[str] = []
        ids: List[str] = []

        # load entire corpus text + ids once; suitable for ~200 pages
        scroll_filter = None
        next_page_offset = None
        while True:
            points, next_page_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                offset=next_page_offset,
                limit=256,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                break
            for p in points:
                payload = p.payload or {}
                text = payload.get("text")
                if not text:
                    continue
                chunk_id = str(payload.get("chunk_id") or p.id)
                texts.append(text)
                ids.append(chunk_id)
            if next_page_offset is None:
                break

        tokenized = [t.split() for t in texts]
        self._bm25_tokens = tokenized
        self._bm25_ids = ids
        self._bm25 = BM25Okapi(tokenized)

    def _dense_search(self, query: str, top_k: int = 20) -> Dict[str, float]:
        q_vec = self.dense_model.get_text_embedding(query)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=q_vec,
            limit=top_k,
            with_payload=True,
        )
        search_result = response.points
        scores: Dict[str, float] = {}
        for rank, p in enumerate(search_result, start=1):
            payload = p.payload or {}
            cid = str(payload.get("chunk_id") or p.id)
            scores[cid] = float(p.score)
        return scores

    def _sparse_search(self, query: str, top_k: int = 100) -> Dict[str, float]:
        self._ensure_bm25()
        assert self._bm25 is not None
        scores = self._bm25.get_scores(query.split())
        scored = list(enumerate(scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]
        dense: Dict[str, float] = {}
        for idx, score in top:
            cid = self._bm25_ids[idx]
            dense[cid] = float(score)
        return dense

    @staticmethod
    def _rrf_fuse(
        dense_ranks: List[str],
        sparse_ranks: List[str],
        k: int = 60,
    ) -> List[Tuple[str, float]]:
        rank_dense: Dict[str, int] = {cid: i + 1 for i, cid in enumerate(dense_ranks)}
        rank_sparse: Dict[str, int] = {cid: i + 1 for i, cid in enumerate(sparse_ranks)}

        all_ids = set(rank_dense) | set(rank_sparse)
        fused: List[Tuple[str, float]] = []
        for cid in all_ids:
            rd = rank_dense.get(cid, len(rank_dense) + k)
            rs = rank_sparse.get(cid, len(rank_sparse) + k)
            score = 1.0 / (k + rd) + 1.0 / (k + rs)
            fused.append((cid, score))
        fused.sort(key=lambda x: x[1], reverse=True)
        return fused

    def _fetch_chunks(self, ids: List[str]) -> Dict[str, Dict]:
        if not ids:
            return {}
        results = self.client.retrieve(
            collection_name=self.collection_name,
            ids=ids,
        )
        out: Dict[str, Dict] = {}
        for p in results:
            payload = p.payload or {}
            cid = str(payload.get("chunk_id") or p.id)
            out[cid] = payload
        return out

    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[RetrievedChunk], List[float], float]:
        """
        Returns top_k reranked chunks and list of reranker scores (for diagnostics),
        plus max_score for confidence gating.
        """
        t0 = time.perf_counter()
        dense_scores = self._dense_search(query, top_k=20)
        t1 = time.perf_counter()
        sparse_scores = self._sparse_search(query)
        t2 = time.perf_counter()

        dense_ranked = sorted(dense_scores.items(), key=lambda x: x[1], reverse=True)
        dense_ids = [cid for cid, _ in dense_ranked]
        sparse_ranked = sorted(sparse_scores.items(), key=lambda x: x[1], reverse=True)
        sparse_ids = [cid for cid, _ in sparse_ranked]

        fused = self._rrf_fuse(dense_ids, sparse_ids, k=60)
        fused_ids = [cid for cid, _ in fused[:20]]

        payloads = self._fetch_chunks(fused_ids)

        pairs = [(query, payloads[cid]["text"]) for cid in fused_ids if cid in payloads]
        rerank_scores = self.reranker.predict(pairs).tolist() if pairs else []
        scored = list(zip(fused_ids, rerank_scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        final_chunks: List[RetrievedChunk] = []
        final_scores: List[float] = []
        for cid, score in scored[:top_k]:
            payload = payloads.get(cid)
            if not payload:
                continue
            text = payload.get("text", "")
            source = payload.get("source_file", "unknown")
            page = int(payload.get("page_number", -1))
            final_chunks.append(
                RetrievedChunk(
                    text=text,
                    source=source,
                    page=page,
                    score=float(score),
                    chunk_id=cid,
                )
            )
            final_scores.append(float(score))

        max_score = max(final_scores) if final_scores else 0.0
        _ = (t0, t1, t2)  # kept to emphasize timing if you want finer logs
        return final_chunks, final_scores, max_score

