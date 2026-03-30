#!/usr/bin/env python3

import argparse
import json
import string
import io
import token_utils
import inverted_index
import constants

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build index")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency of word")
    tf_parser.add_argument("doc_id", type=int, help="Document id")
    tf_parser.add_argument("term", type=str, help="Term")

    idf_parser = subparsers.add_parser("idf", help="Gets idf for given term")
    idf_parser.add_argument("term", type=str, help="Term")

    tfidf_parser = subparsers.add_parser("tfidf", help="Gets tfidf for given term")
    tfidf_parser.add_argument("doc_id", type=int, help="Document id")
    tfidf_parser.add_argument("term", type=str, help="Term")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser(
    "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=constants.BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=constants.BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("limit", type=int, nargs='?', default=5, help="Limits number of results")

    args = parser.parse_args()

    print("Loading movies.")
    with open("data/movies.json", 'r', encoding="utf-8") as file:
        movies = json.load(file)

    invertedIndex = inverted_index.InvertedIndex()
    match args.command:
        case "search":
            print("Searching for: " + args.query)
            invertedIndex.load()
            results = []
            tokenQuery = token_utils.tokenization(args.query)
            for token in tokenQuery:
                docIds = invertedIndex.get_documents(token)
                if len(docIds) > 0:
                    for id in docIds:
                        results.append(invertedIndex.docmap[id])
                        if len(results) == 5:
                            break
                if len(results) == 5:
                    break
            for result in results:
                print(f"{result["id"]} {result["title"]}")
            pass
        case "build":
            invertedIndex.build(movies['movies'])
            invertedIndex.save()
        case "tf":
            invertedIndex.load()
            print(invertedIndex.get_tf(args.doc_id, args.term))
        case "idf":
            invertedIndex.load()
            print(f"Inverse document frequency of '{args.term}': {invertedIndex.get_idf(args.term):.2f}")
        case "tfidf":
            invertedIndex.load()
            tf_idf = invertedIndex.get_tf(args.doc_id, args.term) * invertedIndex.get_idf(args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case "bm25idf":
            invertedIndex.load()
            print(f"BM25 IDF score of '{args.term}': {invertedIndex.get_bm25_idf(args.term) : .2f}")
        case "bm25tf":
            invertedIndex.load()
            bm25tf = invertedIndex.get_bm25_tf(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case "bm25search":
            invertedIndex.load()
            results = invertedIndex.bm25_search(args.query, args.limit)
            for score, movie in results.items():
                print(f"({movie["id"]}) {movie["title"]} - Score: {score:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()