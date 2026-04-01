#!/usr/bin/env python3

import argparse
import lib.semantic_search as ss

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verifies model is loaded")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embeds text")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Creates and verifies embeddings")

    embedquery_parser = subparsers.add_parser("embedquery", help="Embeds query")
    embedquery_parser.add_argument("query", type=str, help="Query to embed")

    search_parser = subparsers.add_parser("search", help="Search for relevant documents")
    search_parser.add_argument("query", type=str, help="Query to search")
    search_parser.add_argument("--limit", type=int, nargs="?", default=5, help="Limit results")

    chunk_parser = subparsers.add_parser("chunk", help="Creates chunks of data")
    chunk_parser.add_argument("text", type=str, help="Position to chunk data")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Size of chunks")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Overlap of chunks")

    args = parser.parse_args()

    match args.command:
        case "verify":
            ss.verify_model()
        case "embed_text":
            ss.embed_text(args.text)
        case "verify_embeddings":
            ss.verify_embeddings()
        case "embedquery":
            ss.embed_query_text(args.query)
        case "search":
            ss.search(args.query, args.limit)
        case "chunk":
            ss.chunk(args.text, args.chunk_size, args.overlap)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()