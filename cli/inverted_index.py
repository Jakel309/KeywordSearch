import token_utils
import pickle
from pathlib import Path
import os
from collections import defaultdict
from collections import Counter
import math
import constants

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths: dict[int, int] = {}
        self.index_path = os.path.join(constants.CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(constants.CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(constants.CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(constants.CACHE_DIR, "doc_lengths.pkl")
    
    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = token_utils.tokenization(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0
        total_doc_length = 0
        for length in self.doc_lengths.values():
            total_doc_length += length
        return total_doc_length / len(self.doc_lengths)

    def get_documents(self, term: str) -> list[int]:
        docs = list(self.index[term.lower()])
        docs.sort()
        return docs

    def build(self, items) -> None:
        for item in items:
            self.__add_document(item['id'], f"{item['title'].lower()} {item['description'].lower()}")
            self.docmap[item['id']] = item
    
    def save(self) -> None:
        if not os.path.isdir(constants.CACHE_DIR):
            os.mkdir(constants.CACHE_DIR)
        with open(self.index_path, 'wb') as file:
            pickle.dump(self.index, file)
        with open(self.docmap_path, 'wb') as file:
            pickle.dump(self.docmap, file)
        with open(self.term_frequencies_path, 'wb') as file:
            pickle.dump(self.term_frequencies, file)
        with open(self.doc_lengths_path, 'wb') as file:
            pickle.dump(self.doc_lengths, file)
    
    def load(self) -> None:
        with open(self.index_path, 'rb') as file:
            self.index = pickle.load(file)
        with open(self.docmap_path, 'rb') as file:
            self.docmap = pickle.load(file)
        with open(self.term_frequencies_path, 'rb') as file:
            self.term_frequencies = pickle.load(file)
        with open(self.doc_lengths_path, 'rb') as file:
            self.doc_lengths = pickle.load(file)
    
    def get_tf(self, doc_id, term) -> int:
        token = token_utils.tokenization(term)
        if len(token) > 1:
            raise ValueError("term must be a single token")
        return self.term_frequencies[doc_id][token[0]]
    
    def get_idf(self, term) -> float:
        token = token_utils.tokenization(term)
        if len(token) > 1:
            raise ValueError("term must be a single token")
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.get_documents(token[0]))
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def get_bm25_idf(self, term: str) -> float:
        token = token_utils.tokenization(term)
        if len(token) > 1:
            raise ValueError("term must be a single token")
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.get_documents(token[0]))
        return math.log((total_doc_count - term_match_doc_count + .5) / (term_match_doc_count + .5) + 1)

    def get_bm25_tf(self, doc_id, term, k1=constants.BM25_K1, b=constants.BM25_B) -> float:
        raw_tf = self.get_tf(doc_id, term)
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        saturation = (raw_tf * (k1 + 1)) / (raw_tf + k1 * length_norm)
        return saturation
    
    def bm25(self, doc_id, term) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)
    
    def bm25_search(self, query, limit) -> dict[int, dict]:
        token_utilsd_query = token_utils.tokenization(query)
        scores: dict[int, int] = {}
        for term in token_utilsd_query:
            docs = self.get_documents(term)
            for doc in docs:
                bm25 = self.bm25(doc, term)
                if doc in scores:
                    scores[doc] += bm25
                else:
                    scores[doc] = bm25
        sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
        top_results = {k: v for k, v in list(sorted_scores.items())[:limit]}
        docs_to_return: dict[int, dict] = {}
        for key in top_results:
            docs_to_return[top_results[key]] = self.docmap[key]
        return docs_to_return