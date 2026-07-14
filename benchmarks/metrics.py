import math
from statistics import mean


def recall_at_k(
    retrieved_documents,
    relevant_documents,
    k,
):
    relevant_ids = {
        document_id
        for document_id, score in relevant_documents.items()
        if score > 0
    }

    if not relevant_ids:
        return 0.0

    retrieved_ids = set(retrieved_documents[:k])

    relevant_retrieved = retrieved_ids.intersection(relevant_ids)

    return len(relevant_retrieved) / len(relevant_ids)


def reciprocal_rank_at_k(
    retrieved_documents,
    relevant_documents,
    k,
):
    relevant_ids = {
        document_id
        for document_id, score in relevant_documents.items()
        if score > 0
    }

    for rank, document_id in enumerate(
        retrieved_documents[:k],
        start=1,
    ):
        if document_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def dcg_at_k(
    retrieved_documents,
    relevant_documents,
    k,
):
    score = 0.0

    for rank, document_id in enumerate(
        retrieved_documents[:k],
        start=1,
    ):
        relevance = relevant_documents.get(document_id, 0)

        if relevance > 0:
            score += (
                (2**relevance - 1)
                / math.log2(rank + 1)
            )

    return score


def ndcg_at_k(
    retrieved_documents,
    relevant_documents,
    k,
):
    actual_dcg = dcg_at_k(
        retrieved_documents,
        relevant_documents,
        k,
    )

    ideal_relevances = sorted(
        relevance
        for relevance in relevant_documents.values()
        if relevance > 0
    )[:k]

    ideal_relevances.reverse()

    ideal_dcg = sum(
        (2**relevance - 1)
        / math.log2(rank + 1)
        for rank, relevance in enumerate(
            ideal_relevances,
            start=1,
        )
    )

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def calculate_metrics(
    retrieved_documents,
    relevant_documents,
    top_k,
):
    return {
        f"recall@{top_k}": recall_at_k(
            retrieved_documents,
            relevant_documents,
            top_k,
        ),
        f"mrr@{top_k}": reciprocal_rank_at_k(
            retrieved_documents,
            relevant_documents,
            top_k,
        ),
        f"ndcg@{top_k}": ndcg_at_k(
            retrieved_documents,
            relevant_documents,
            top_k,
        ),
    }


def average_metrics(metrics):
    if not metrics:
        return {}

    return {
        metric_name: mean(
            result[metric_name]
            for result in metrics
        )
        for metric_name in metrics[0]
    }
