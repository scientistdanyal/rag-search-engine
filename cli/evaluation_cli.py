import argparse
import json
import os

from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "golden_dataset.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )
    args = parser.parse_args()
    limit = args.limit

    with open(GOLDEN_DATASET_PATH, "r") as f:
        golden_dataset = json.load(f)

    documents = load_movies()
    searcher = HybridSearch(documents)

    print(f"k={limit}\n")

    for test_case in golden_dataset["test_cases"]:
        query = test_case["query"]
        relevant_docs = test_case["relevant_docs"]
        relevant_set = set(relevant_docs)

        results = searcher.rrf_search(query, k=60, limit=limit)
        retrieved_titles = [r["title"] for r in results]

        relevant_retrieved = sum(
            1 for title in retrieved_titles if title in relevant_set
        )
        precision = (
            relevant_retrieved / len(retrieved_titles) if retrieved_titles else 0.0
        )

        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved_titles)}")
        print(f"  - Relevant: {', '.join(relevant_docs)}")
        print()


if __name__ == "__main__":
    main()