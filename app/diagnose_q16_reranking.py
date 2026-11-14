"""Compare vector-search candidates with cross-encoder reranking for one question."""

from sentence_transformers import CrossEncoder

from app.chunker import chunk_document
from app.document_loader import load_documents
from app.embedder import create_embedder, embed_chunks
from app.reranker import DEFAULT_RERANKER_MODEL, rerank_chunks
from app.vector_search import search_chunks

QUESTION = (
    "My Kubernetes pod shows ImagePullBackOff. "
    "How do I diagnose and fix the image pull failure?"
)
FINAL_TOP_K = 3


def main():
    documents = load_documents()
    chunks = []

    for document in documents:
        chunks.extend(chunk_document(document))

    embedding_model = create_embedder()
    embedded_chunks = embed_chunks(chunks, embedding_model)
    reranker = CrossEncoder(DEFAULT_RERANKER_MODEL)

    for candidate_count in (10, 20):
        candidates = search_chunks(
            query=QUESTION,
            embedded_chunks=embedded_chunks,
            model=embedding_model,
            top_k=candidate_count,
            min_score=-1.0,
        )

        reranked = rerank_chunks(
            query=QUESTION,
            chunks=candidates,
            model=reranker,
            top_k=FINAL_TOP_K,
        )

        print(f"\nQ16: CANDIDATE POOL = {candidate_count}")
        print("=" * 65)

        print("\nVector-search candidates:")
        for rank, chunk in enumerate(candidates, start=1):
            print(
                f"{rank:2}. {chunk['source']} | "
                f"{chunk['section']} | "
                f"vector={chunk['score']:.4f}"
            )

        print("\nFinal reranked top 3:")
        for rank, chunk in enumerate(reranked, start=1):
            print(
                f"{rank}. {chunk['source']} | "
                f"{chunk['section']} | "
                f"vector={chunk['score']:.4f} | "
                f"cross-encoder={chunk['rerank_score']:.4f}"
            )


if __name__ == "__main__":
    main()