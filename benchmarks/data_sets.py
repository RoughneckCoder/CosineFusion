from datasets import load_dataset


DATASET_NAME = "BeIR/scifact"
QRELS_DATASET_NAME = "BeIR/scifact-qrels"


def load_scifact():
    print("Loading SciFact...")

    corpus_dataset = load_dataset(
        DATASET_NAME,
        "corpus",
        split="corpus",
    )

    queries_dataset = load_dataset(
        DATASET_NAME,
        "queries",
        split="queries",
    )

    qrels_dataset = load_dataset(
        QRELS_DATASET_NAME,
        split="test",
    )

    documents = {
        str(row["_id"]): (
            f"{row.get('title', '')} {row['text']}"
        ).strip()
        for row in corpus_dataset
    }

    queries = {
        str(row["_id"]): row["text"]
        for row in queries_dataset
    }

    relevance = {}

    for row in qrels_dataset:
        query_id = str(row["query-id"])
        document_id = str(row["corpus-id"])
        score = int(row["score"])

        relevance.setdefault(query_id, {})[document_id] = score

    return documents, queries, relevance
