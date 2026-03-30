#!/usr/bin/env python3

import argparse
import lib.semantic_search as ss

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verifies model is loaded")

    args = parser.parse_args()

    match args.command:
        case "verify":
            ss.verify_model()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()