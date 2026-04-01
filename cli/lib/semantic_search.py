from sentence_transformers import SentenceTransformer
import numpy as np
import constants
import os
import json

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.embeddings_file = constants.CACHE_DIR + "/movie_embeddings.npy"
    
    def generate_embeddings(self, text: str) -> Tensor :
        if text.strip() == "":
            raise ValueError("Text must not be blank")
        
        embedding = self.model.encode([text])[0]
        return embedding

    def build_embeddings(self, documents: list({})) -> Tensor:
        self.documents = documents

        doc_strings = []
        for document in documents:
            self.document_map[document["id"]] = document
            doc_strings.append(f"{document['title']}: {document['description']}")
        
        self.embeddings = self.model.encode(doc_strings, show_progress_bar=True)

        with open(self.embeddings_file, 'wb') as f:
            np.save(f, self.embeddings)
        
        return self.embeddings
    
    def load_or_create_embeddings(self, documents) -> None:
        self.documents = documents
        doc_strings = []
        for document in documents:
            self.document_map[document["id"]] = document
            doc_strings.append(f"{document['title']}: {document['description']}")
        
        if os.path.exists(self.embeddings_file):
            with open(self.embeddings_file, 'rb') as f:
                embeddings = np.load(f)
            if len(embeddings) == len(documents):
                self.embeddings = embeddings
                return
        self.build_embeddings(documents)

    def search(self, query: str, limit: int):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        
        embedding = self.generate_embeddings(query)
        scores = []
        count = 0
        for embed in self.embeddings:
            score = cosine_similarity(embedding, embed)
            scores.append((score, self.documents[count]))
            count += 1
        sorted_scores = sorted(scores, key=lambda tup: tup[0], reverse=True)
        docs_to_return = []
        for i in range(limit):
            doc = {}
            doc["score"] = sorted_scores[i][0]
            doc["title"] = sorted_scores[i][1]["title"]
            doc["description"] = sorted_scores[i][1]["description"]
            docs_to_return.append(doc)
        
        return docs_to_return


def verify_model() -> None:
    search = SemanticSearch()
    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")

def embed_text(text: str) -> None:
    search = SemanticSearch()
    embedding = search.generate_embeddings(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings() -> None:
    search = SemanticSearch()
    with open("data/movies.json", 'r', encoding="utf-8") as file:
        movies = json.load(file)
    search.load_or_create_embeddings(movies["movies"])
    print(f"Number of docs: {len(movies)}")
    print(f"Embeddings shape: {search.embeddings.shape[0]} vectors in {search.embeddings.shape[1]} dimensions")

def embed_query_text(query: str) -> None:
    search = SemanticSearch()
    embedding = search.generate_embeddings(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1: list(int), vec2: list(int)):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def search(query, limit):
    search = SemanticSearch()
    with open("data/movies.json", 'r', encoding="utf-8") as file:
        movies = json.load(file)
    search.load_or_create_embeddings(movies["movies"])
    
    results = search.search(query, limit)
    count = 1
    for result in results:
        print(f"{count}. {result["title"]}: (score:{result["score"]}\n{result["description"]}")
        count += 1

def chunk(text, size, overlap):
    split_text = text.split(' ')
    chunks = []
    chunk = ""
    count = 0
    for word in split_text:
        chunk += word + " "
        count += 1
        if count == size:
            if overlap > 0 and len(chunks) > 0:
                last_chunk = chunks[len(chunks) - 1].split(" ")
                chunk = " ".join(last_chunk[len(last_chunk)-overlap:]) + " " + chunk
            chunks.append(chunk.strip())
            count = 0
            chunk = ""
    if chunk != "":
        if overlap > 0 and len(chunks) > 0:
            last_chunk = chunks[len(chunks) - 1].split(" ")
            chunk = " ".join(last_chunk[len(last_chunk)-overlap:]) + " " + chunk
        chunks.append(chunk)
    
    print(f"Chunking {len(text)} characters")
    count = 1
    for chunk in chunks:
        print(f"{count}. {chunk}")
        count += 1