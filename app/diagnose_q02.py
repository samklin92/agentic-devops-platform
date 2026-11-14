from sentence_transformers import CrossEncoder

from app.chunker import chunk_document
from app.document_loader import load_documents
from app.embedder import create_embedder, embed_chunks
from app.reranker import DEFAULT_RERANKER_MODEL
from app.vector_search import search_chunks


QUESTION = "A container was killed with exit code 137. What does that tell me?"


def main():
    print("Loading knowledge base...")

    documents = load_documents()
    chunks = []

    for document in documents:
        chunks.extend(chunk_document(document))

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")

    print("\nLoading embedding model...")
    embedding_model = create_embedder()

    print("Embedding document chunks...")
    embedded_chunks = embed_chunks(chunks, embedding_model)

    print("\nRunning vector search...")
    candidates = search_chunks(
        query=QUESTION,
        embedded_chunks=embedded_chunks,
        model=embedding_model,
        top_k=10,
        min_score=-1.0,
    )

    print("\nLoading cross-encoder...")
    reranker = CrossEncoder(DEFAULT_RERANKER_MODEL)

    pairs = [(QUESTION, chunk["content"]) for chunk in candidates]
    scores = reranker.predict(pairs, show_progress_bar=False)

    reranked = sorted(
        zip(candidates, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    print("\n" + "=" * 70)
    print("BASELINE VECTOR SEARCH — TOP 10")
    print("=" * 70)

    for rank, chunk in enumerate(candidates, start=1):
        print(f"\nRank: {rank}")
        print(f"Source: {chunk['source']}")
        print(f"Section: {chunk['section']}")
        print(f"Vector similarity: {chunk['score']:.4f}")
        print(f"Content: {chunk['content'][:450]}")

    print("\n" + "=" * 70)
    print("CROSS-ENCODER RERANKING — TOP 10")
    print("=" * 70)

    for rank, (chunk, score) in enumerate(reranked, start=1):
        print(f"\nRank: {rank}")
        print(f"Source: {chunk['source']}")
        print(f"Section: {chunk['section']}")
        print(f"Vector similarity: {chunk['score']:.4f}")
        print(f"Cross-encoder score: {float(score):.4f}")
        print(f"Content: {chunk['content'][:450]}")


if __name__ == "__main__":
    main()
