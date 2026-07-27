import argparse
from ast import Pass
from lib.sement_search import (
    chunk_text,
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    semantic_search,
)


def main()->None:

    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify that embedding model is loaded")

    single_embed_parser = subparsers.add_parser("embed_text", help="Embed text into embedding model")
    single_embed_parser.add_argument("text", type=str, help="Text to be embedded")
    subparsers.add_parser("verify_embeddings", help="Verify embeddings for the movie dataset")

    embed_query_parser = subparsers.add_parser("embed_query", help="Generate an embedding for a search query")
    embed_query_parser.add_argument("query", type=str, help="Text to be embedded")
    search_parser = subparsers.add_parser("search", help="Search query")
    search_parser.add_argument("query", type=str, help="Text to be embedded")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")


    chunk_parser = subparsers.add_parser("chunk", help="Split text into fixed-size chunks")
    chunk_parser.add_argument(
        "--chunk-size", type=int, default=200, help="Size of each chunk"
    )

    args = parser.parse_args()



    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            semantic_search(args.query, args.limit)

        case "chunk":
            chunk_text(args.text, args.chunk_size)

        case _:
            parser.print_help()




if __name__ == "__main__":
    main()