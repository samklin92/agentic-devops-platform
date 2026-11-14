from sentence_transformers import CrossEncoder

from app.evaluate_reranking import (
    FINAL_TOP_K,
    MIN_SCORE,
    RERANKER_MODEL,
    build_knowledge_base,
    load_questions,
    load_ranking_questions,
)
from app.embedder import create_embedder, embed_chunks
from app.ranking_evaluator import reciprocal_rank
from app.reranker import rerank_chunks
from app.vector_search import search_chunks


def main():
    questions = load_questions()
    labels = load_ranking_questions()
    ranking_questions = [q for q in questions if q["id"] in labels]

    chunks = build_knowledge_base()
    embedder = create_embedder()
    embedded = embed_chunks(chunks, embedder)
    reranker = CrossEncoder(RERANKER_MODEL)

    print("Candidate pool | Baseline MRR | Reranked MRR | Regressions")
    print("-" * 65)

    for candidate_count in (10, 20):
        baseline_ranks = []
        reranked_ranks = []
        regressions = 0

        for question in ranking_questions:
            candidates = search_chunks(
                query=question["question"],
                embedded_chunks=embedded,
                model=embedder,
                top_k=candidate_count,
                min_score=MIN_SCORE,
            )
            relevant = labels[question["id"]]
            baseline = candidates[:FINAL_TOP_K]
            reranked = rerank_chunks(
                query=question["question"],
                chunks=candidates,
                model=reranker,
                top_k=FINAL_TOP_K,
            )
            baseline_rr = reciprocal_rank(baseline, relevant)
            reranked_rr = reciprocal_rank(reranked, relevant)
            baseline_ranks.append(baseline_rr)
            reranked_ranks.append(reranked_rr)
            if reranked_rr < baseline_rr:
                regressions += 1

        baseline_mrr = sum(baseline_ranks) / len(baseline_ranks)
        reranked_mrr = sum(reranked_ranks) / len(reranked_ranks)
        print(
            f"{candidate_count:14} | {baseline_mrr:12.4f} | "
            f"{reranked_mrr:12.4f} | {regressions}"
        )


if __name__ == "__main__":
    main()
