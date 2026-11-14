from app.chunker import chunk_document
from app.document_loader import load_documents
from app.embedder import create_embedder, embed_chunks
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

    model = create_embedder()
    embedded_chunks = embed_chunks(chunks, model)

    results = search_chunks(
        query=QUESTION,
        embedded_chunks=embedded_chunks,
        model=model,
        top_k=5,
        min_score=-1.0,
    )

    for rank, result in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print(f"Source: {result['source']}")
        print(f"Section: {result['section']}")
        print(f"Similarity: {result['score']:.4f}")
        print(f"Content: {result['content'][:350]}")


if __name__ == "__main__":
    main()
