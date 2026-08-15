import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query: str, k: int = 60, limit: int = 10) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        combined: dict[str, dict] = {}

        for rank, result in enumerate(bm25_results, start=1):
            doc_id = str(result["id"])
            combined[doc_id] = {
                "id": result["id"],
                "title": result["title"],
                "document": result["document"],
                "bm25_rank": rank,
                "semantic_rank": None,
                "rrf_score": rrf_score(rank, k),
            }

        for rank, result in enumerate(semantic_results, start=1):
            doc_id = str(result["id"])
            if doc_id not in combined:
                combined[doc_id] = {
                    "id": result["id"],
                    "title": result["title"],
                    "document": result["document"],
                    "bm25_rank": None,
                    "semantic_rank": rank,
                    "rrf_score": rrf_score(rank, k),
                }
            else:
                combined[doc_id]["semantic_rank"] = rank
                combined[doc_id]["rrf_score"] += rrf_score(rank, k)

        ranked = sorted(
            combined.values(),
            key=lambda d: d["rrf_score"],
            reverse=True,
        )
        return ranked[:limit]
    
    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        bm25_scores = normalize_scores([r["score"] for r in bm25_results])
        semantic_scores = normalize_scores([r["score"] for r in semantic_results])

        combined: dict[str, dict] = {}

        for result, score in zip(bm25_results, bm25_scores):
            doc_id = str(result["id"])
            combined[doc_id] = {
                "id": result["id"],
                "title": result["title"],
                "document": result["document"],
                "bm25_score": score,
                "semantic_score": 0.0,
            }

        for result, score in zip(semantic_results, semantic_scores):
            doc_id = str(result["id"])
            if doc_id not in combined:
                combined[doc_id] = {
                    "id": result["id"],
                    "title": result["title"],
                    "document": result["document"],
                    "bm25_score": 0.0,
                    "semantic_score": score,
                }
            else:
                combined[doc_id]["semantic_score"] = score

        for doc in combined.values():
            doc["hybrid_score"] = hybrid_score(
                doc["bm25_score"],
                doc["semantic_score"],
                alpha,
            )

        ranked = sorted(
            combined.values(),
            key=lambda d: d["hybrid_score"],
            reverse=True,
        )
        return ranked[:limit]







def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0] * len(scores)

    return [(s - min_score) / (max_score - min_score) for s in scores]

def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score






def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)