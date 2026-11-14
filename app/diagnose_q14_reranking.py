from sentence_transformers import CrossEncoder

from app.chunker import chunk_document
from app.document_loader import load_documents
from app.embedder import create_embedder, embed_chunks
from app.reranker import DEFAULT_RERANKER_MODEL
from app.vector_search import search_chunks


QUESTION = (
    "My application has database connection timeouts after a deployment. "
    "What configuration and connectivity checks should I perform?"
)


def main():
    documents = load_documents()
    chunks = []

    for document in documents:
        chunks.extend(chunk_document(document))

    embedding_model = create_embedder()
    embedded_chunks = embed_chunks(chunks, embedding_model)

    candidates = search_chunks(
        query=QUESTION,
        embedded_chunks=embedded_chunks,
        model=embedding_model,
        top_k=10,
        min_score=-1.0,
    )

    reranker = CrossEncoder(DEFAULT_RERANKER_MODEL)
    pairs = [(QUESTION, chunk["content"]) for chunk in candidates]
    scores = reranker.predict(pairs, show_progress_bar=False)

    scored_candidates = sorted(
        zip(candidates, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    print("\nCROSS-ENCODER RANKING FOR Q14")
    print("=" * 60)

    for rank, (chunk, score) in enumerate(scored_candidates, start=1):
        print(f"\nRank: {rank}")
        print(f"Section: {chunk['section']}")
        print(f"Vector similarity: {chunk['score']:.4f}")
        print(f"Cross-encoder score: {float(score):.4f}")
        print(f"Content: {chunk['content'][:500]}")


if __name__ == "__main__":
    main()
