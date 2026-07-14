import json
import time
from pathlib import Path
from statistics import mean

import numpy as np
from sentence_transformers import SentenceTransformer

from data_sets import load_scifact
from metrics import calculate_metrics, average_metrics


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 10

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_FILE = RESULTS_DIR / "scifact.json"


def cosine_search(
    query_embedding,
    document_embeddings,
    document_ids,
    top_k,
):
    scores = document_embeddings @ query_embedding

    top_indices = np.argpartition(
        scores,
        -top_k,
    )[-top_k:]

    ranked_indices = top_indices[
        np.argsort(scores[top_indices])[::-1]
    ]

    return [
        document_ids[index]
        for index in ranked_indices
    ]


def benchmark():
    documents, queries, relevance = load_scifact()

    print(f"Documents: {len(documents):,}")
    print(f"Queries: {len(relevance):,}")

    print(f"\nLoading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    document_ids = list(documents.keys())
    document_texts = list(documents.values())

    print("\nEncoding document corpus...")

    document_embeddings = model.encode(
        document_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    baseline_metrics = []
    baseline_latencies = []

    print("\nRunning cosine baseline...")

    total_queries = len(relevance)

    for query_number, (
        query_id,
        relevant_documents,
    ) in enumerate(relevance.items(), start=1):

        query_text = queries[query_id]

        query_embedding = model.encode(
            query_text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        start_time = time.perf_counter()

        retrieved_documents = cosine_search(
            query_embedding,
            document_embeddings,
            document_ids,
            TOP_K,
        )

        elapsed_time = time.perf_counter() - start_time

        baseline_latencies.append(elapsed_time)

        baseline_metrics.append(
            calculate_metrics(
                retrieved_documents,
                relevant_documents,
                TOP_K,
            )
        )

        if query_number % 50 == 0:
            print(
                f"Processed "
                f"{query_number}/{total_queries} queries"
            )

    baseline_results = average_metrics(
        baseline_metrics,
    )

    baseline_results["mean_latency_ms"] = (
        mean(baseline_latencies) * 1000
    )

    results = {
        "dataset": "BeIR/scifact",
        "embedding_model": MODEL_NAME,
        "top_k": TOP_K,
        "document_count": len(documents),
        "query_count": len(relevance),
        "baseline_cosine": baseline_results,
    }

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
        )

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    print("\nStandard Cosine Similarity")

    for metric_name, value in baseline_results.items():
        if metric_name == "mean_latency_ms":
            print(f"{metric_name}: {value:.3f} ms")
        else:
            print(f"{metric_name}: {value:.4f}")

    print(f"\nResults written to: {RESULTS_FILE}")


if __name__ == "__main__":
    benchmark()
