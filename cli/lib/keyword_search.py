from .search_utils import (
    load_movies, 
    DEAFAULT_SEARCH_LIMIT, 
    load_stopwords, 
    CACHE_DIR,
    TERM_FREQUENCIES_PATH,
    )
import string
from nltk.stem import PorterStemmer
import pickle
from collections import defaultdict, Counter
import os
import math



class InvertedIndex:

    def __init__(self):
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies = defaultdict(Counter)


    
    
    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            doc_id = m['id']
            doc_description = f"{m['title']} {m['description']}"
            self.__add_document( doc_id, doc_description)
            self.docmap[doc_id] = m
        

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, 'wb') as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, 'wb') as f:
            pickle.dump(self.docmap, f)
        
        with open(TERM_FREQUENCIES_PATH, 'wb') as f:
            pickle.dump(self.term_frequencies, f)

    
    def load(self) -> None:
        with open(self.index_path, 'rb') as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, 'rb') as f:
            self.docmap = pickle.load(f)

        with open(TERM_FREQUENCIES_PATH, 'rb') as f:
            self.term_frequencies = pickle.load(f)



    def get_documents(self, term: str)->list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def __add_document(self, doc_id: int, text: str)->None:
        tokens = tokenize_text(text)
        for token in tokens:
            self.index[token].add(doc_id) 
        self.term_frequencies[doc_id].update(tokens)    


    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        
        token = tokens[0]
        return self.term_frequencies[doc_id][token]


    def get_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count) / (term_doc_count + 1))

    def get_tf_idf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf
    
def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()




def search_command(query: str, limit: int = DEAFAULT_SEARCH_LIMIT)->list[dict]:

    movies = load_movies()
    idx = InvertedIndex()
    idx.load()
    query_tokens = tokenize_text(query)
    seen, results = set(), []
    for query_token in query_tokens:
        matching_doc_ids = idx.get_documents(query_token)
        for doc_id in matching_doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = idx.docmap[doc_id]
            results.append(doc)
            if len(results) >= limit:
                return results
    return results
        






def text_preprocessing(text: str)->list[str]:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text




def tokenize_text(text: str) -> list[str]:
    text = text_preprocessing(text)
    tokens = text.split()
    valid_tokens = [token for token in tokens if token]

    stop_words = load_stopwords()
    valid_words = []
    for token in valid_tokens:
        if token not in stop_words:
            valid_words.append(token)
    
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in valid_words]
    return stemmed_words



def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, term)




def idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(term)


def tfidf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf_idf(doc_id, term)
    