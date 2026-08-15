import json
import os
from typing import Any, TypedDict

SCORE_PRECISION = 3
DEFAULT_SEARCH_LIMIT = 5
DEAFAULT_SEARCH_LIMIT = DEFAULT_SEARCH_LIMIT  # alias for older imports
BM25_K1 = 1.5
BM25_B = 0.75

DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 0
DEFAULT_SEMANTIC_CHUNK_SIZE = 4


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOP_WORDS = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

TERM_FREQUENCIES_PATH = os.path.join(PROJECT_ROOT, "cache", "term_frequencies.pkl")
MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")
CHUNK_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
CHUNK_METADATA_PATH = os.path.join(CACHE_DIR, "chunk_metadata.json")


class Movie(TypedDict):
    id: int
    title: str
    description: str


def format_search_result(
    doc_id: str, title: str, document: str, score: float, **metadata: Any
) -> dict[str, Any]:
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }

def load_stopwords()->list[str]:
    with open(STOP_WORDS, "r") as file:
        return file.read().splitlines()

def load_movies()->list[dict]:
    with open(DATA_PATH, "r") as file:
        data = json.load(file)
    return data["movies"]