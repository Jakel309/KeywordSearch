#!/usr/bin/env python3

import argparse
import json
import string
import io
import tokenize
import inverted_index

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
            tokenQuery = tokenize.tokenization(args.query)
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
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()