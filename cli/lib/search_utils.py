import json
import os


DEAFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOP_WORDS = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")


def load_stopwords()->list[str]:
    with open(STOP_WORDS, "r") as file:
        return file.read().splitlines()

def load_movies()->list[dict]:
    with open(DATA_PATH, "r") as file:
        data = json.load(file)
    return data["movies"]