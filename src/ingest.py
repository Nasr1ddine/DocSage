import os
import time
import uuid
from dataclasses import dataclass
from typing import List, Literal, Tuple

try:
    from dotenv import load_dotenv  # type: ignore[import]
except Exception:  # pragma: no cover
    def load_dotenv(*args, **kwargs):
        return None

from qdrant_client import QdrantClient, models as qmodels  # type: ignore[import]

from llama_index.core import SimpleDirectoryReader  # type: ignore[import]
from llama_index.core.node_parser import SentenceSplitter  # type: ignore[import]
from llama_index.embeddings.openai import OpenAIEmbedding  # type: ignore[import]


load_dotenv()


StrategyName = Literal["naive", "tuned"]


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict


def get_qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY") or None
    return QdrantClient(url=url, api_key=api_key)


def get_collection_name(strategy: StrategyName) -> str:
    if strategy == "naive":
        return os.getenv("QDRANT_COLLECTION_NAIVE", "docsage_naive")
    return os.getenv("QDRANT_COLLECTION_TUNED", "docsage_tuned")


def get_embedding_model() -> OpenAIEmbedding:
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    return OpenAIEmbedding(model=model)


def _ensure_collection(client: QdrantClient, collection_name: str, dim: int) -> None:
    exists = False
    try:
        client.get_collection(collection_name)
        exists = True
    except Exception:
        exists = False

    if not exists:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=dim,
                distance=qmodels.Distance.COSINE,
            ),
        )


def _load_documents(data_dir: str) -> List:
    reader = SimpleDirectoryReader(data_dir, required_exts=[".pdf"], filename_as_id=True)
    return reader.load_data()


def _chunk_naive(documents: List) -> List[Chunk]:
    chunks: List[Chunk] = []
    chunk_size = 1000
    for doc in documents:
        text = doc.text
        source_file = doc.metadata.get("file_name") or doc.metadata.get("filename")
        # naive: no metadata except minimal source to help debugging
        for i in range(0, len(text), chunk_size):
            chunk_text = text[i : i + chunk_size]
            if not chunk_text.strip():
                continue
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    text=chunk_text,
                    metadata={"source_file": source_file} if source_file else {},
                )
            )
    return chunks


def _extract_page_number(node_metadata: dict) -> int:
    page = node_metadata.get("page_label") or node_metadata.get("page_number")
    if isinstance(page, int):
        return page
    try:
        return int(page)
    except Exception:
        return -1


def _heuristic_section_heading(text: str) -> str:
    # simple heuristic: look for common headings in the first 300 characters
    window = text[:300].lower()
    if "risk factors" in window:
        return "Risk Factors"
    if "management's discussion" in window:
        return "Management Discussion & Analysis"
    if "conclusion" in window:
        return "Conclusion"
    if "introduction" in window:
        return "Introduction"
    return "Unknown"


def _chunk_tuned(documents: List) -> List[Chunk]:
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=int(0.1 * 512))
    chunks: List[Chunk] = []

    for doc in documents:
        source_file = doc.metadata.get("file_name") or doc.metadata.get("filename")
        page_number = _extract_page_number(doc.metadata)
        split_nodes = splitter.get_nodes_from_documents([doc])
        for idx, node in enumerate(split_nodes):
            text = node.text
            if not text.strip():
                continue
            section_heading = _heuristic_section_heading(text)
            metadata = {
                "source_file": source_file,
                "page_number": page_number,
                "chunk_index": idx,
                "section_heading": section_heading,
            }
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    text=text,
                    metadata=metadata,
                )
            )
    return chunks


def _embed_chunks(chunks: List[Chunk], embed_model: OpenAIEmbedding) -> List[List[float]]:
    texts = [c.text for c in chunks]
    embeddings = embed_model.get_text_embedding_batch(texts)
    # sanity check: print example chunk + metadata to verify splits
    if chunks:
        sample = chunks[min(3, len(chunks) - 1)]
        print("\n=== Sample embedded chunk ===")
        print(sample.text[:400].replace("\n", " ") + ("..." if len(sample.text) > 400 else ""))
        print("Metadata:", sample.metadata)
        print("=== End sample ===\n")
    return embeddings


def _upsert_qdrant(
    client: QdrantClient,
    collection_name: str,
    chunks: List[Chunk],
    embeddings: List[List[float]],
) -> None:
    points = []
    for chunk, emb in zip(chunks, embeddings):
        payload = {
            "text": chunk.text,
            "chunk_id": chunk.id,
        }
        payload.update(chunk.metadata or {})
        points.append(
            qmodels.PointStruct(
                id=chunk.id,
                vector=emb,
                payload=payload,
            )
        )

    client.upsert(collection_name=collection_name, points=points)


def _estimate_storage_mb(chunks: List[Chunk], dim: int) -> float:
    text_bytes = sum(len(c.text.encode("utf-8")) for c in chunks)
    vector_bytes = len(chunks) * dim * 4  # float32
    total_mb = (text_bytes + vector_bytes) / (1024 * 1024)
    return round(total_mb, 3)


def run_ingest(strategy: StrategyName) -> Tuple[int, float, float]:
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw_pdfs")
    documents = _load_documents(data_dir)
    if not documents:
        raise RuntimeError(f"No PDFs found in {data_dir}. Please add 5–10 PDFs first.")

    if strategy == "naive":
        chunks = _chunk_naive(documents)
    else:
        chunks = _chunk_tuned(documents)

    embed_model = get_embedding_model()
    # derive vector dimension from a sample embedding to avoid relying on private APIs
    sample_vec = embed_model.get_text_embedding("dimension probe")
    dim = len(sample_vec)

    collection_name = get_collection_name(strategy)
    client = get_qdrant_client()
    _ensure_collection(client, collection_name, dim)

    t0 = time.perf_counter()
    embeddings = _embed_chunks(chunks, embed_model)
    t1 = time.perf_counter()
    _upsert_qdrant(client, collection_name, chunks, embeddings)
    t2 = time.perf_counter()

    storage_mb = _estimate_storage_mb(chunks, dim)
    print(
        f"Ingestion ({strategy}) complete: {len(chunks)} chunks, "
        f"storage≈{storage_mb} MB, embed_time={t1 - t0:.2f}s, upsert_time={t2 - t1:.2f}s"
    )
    return len(chunks), storage_mb, t2 - t0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest PDFs into Qdrant with different chunking strategies.")
    parser.add_argument("--strategy", choices=["naive", "tuned"], default="naive")
    args = parser.parse_args()

    count, storage_mb, duration = run_ingest(args.strategy)  # noqa: F841

