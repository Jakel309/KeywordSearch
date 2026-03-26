import tokenize
import pickle
from pathlib import Path
import os
from collections import defaultdict
from collections import Counter
import math

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = {}
    
    def __add_document(self, doc_id: int, text: str) -> None:
        tokenText = tokenize.tokenization(text)
        for token in tokenText:
            self.index[token].add(doc_id)
            if doc_id not in self.term_frequencies:
                self.term_frequencies[doc_id] = Counter()
            self.term_frequencies[doc_id][token] += 1
            

    def get_documents(self, term: str) -> list[int]:
        docs = list(self.index[term.lower()])
        docs.sort()
        return docs

    def build(self, items) -> None:
        for item in items:
            self.__add_document(item['id'], f"{item['title'].lower()} {item['description'].lower()}")
            self.docmap[item['id']] = item
    
    def save(self) -> None:
        if not os.path.isdir('cache'):
            os.mkdir("cache")
        with open("cache/index.pkl", 'wb') as file:
            pickle.dump(self.index, file)
        with open("cache/docmap.pkl", 'wb') as file:
            pickle.dump(self.docmap, file)
        with open("cache/term_frequencies.pkl", 'wb') as file:
            pickle.dump(self.term_frequencies, file)
    
    def load(self) -> None:
        with open("cache/index.pkl", 'rb') as file:
            self.index = pickle.load(file)
        with open("cache/docmap.pkl", 'rb') as file:
            self.docmap = pickle.load(file)
        with open("cache/term_frequencies.pkl", 'rb') as file:
            self.term_frequencies = pickle.load(file)
    
    def get_tf(self, doc_id, term) -> int:
        token = tokenize.tokenization(term)
        if len(token) > 1:
            return 0
        return self.term_frequencies[doc_id][term]
    
    def get_idf(self, term) -> float:
        token = tokenize.tokenization(term)
        if len(token) > 1:
            return 0
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.get_documents(token[0]))
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def get_bm25_idf(self, term: str) -> float:
        token = tokenize.tokenization(term)
        if len(token) > 1:
            return 0
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.get_documents(token[0]))
        return math.log((total_doc_count - term_match_doc_count + .5) / (term_match_doc_count + .5) + 1)
