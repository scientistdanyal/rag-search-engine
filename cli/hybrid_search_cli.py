import argparse
from lib.hybrid_search import HybridSearch, normalize_scores
from lib.search_utils import load_movies
from llm import enhance_query_spell, enhance_query_rewrite, enhance_query_expand, rerank_individual, rerank_batch, rerank_cross_encoder

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize a list of scores using min-max normalization"
    )
    normalize_parser.add_argument(
        "scores",
        nargs="*",
        type=float,
        help="List of scores to normalize",
    )

    weighted_parser = subparsers.add_parser(
        "weighted-search",
        help="Hybrid search with weighted BM25 + semantic scores",
    )
    weighted_parser.add_argument("query", type=str, help="Search query")
    weighted_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for BM25 vs semantic (default: 0.5)",
    )
    weighted_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to return",
    )


    rrf_parser = subparsers.add_parser(
    "rrf-search",
    help="Hybrid search using Reciprocal Rank Fusion",
    )
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument(
        "-k",
        type=int,
        default=60,
        help="RRF constant k (default: 60)",
    )
    rrf_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to return",
    )

    rrf_parser.add_argument(
    "--enhance",
    type=str,
    choices=["spell", "rewrite", "expand"],
    help="Query enhancement method: spell (fix typos) or rewrite (improve clarity)",
)
    rrf_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Rerank method: individual (rerank each document individually), batch (rerank documents in batches), cross_encoder (rerank documents using cross-encoder)"
    )
    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized = normalize_scores(args.scores)
            for score in normalized:
                print(f"* {score:.4f}")

        case "weighted-search":
            documents = load_movies()
            searcher = HybridSearch(documents)
            results = searcher.weighted_search(
                args.query, args.alpha, args.limit
            )
            for i, result in enumerate(results, start=1):
                print(f"{i}. {result['title']}")
                print(f"  Hybrid Score: {result['hybrid_score']:.3f}")
                print(
                    f"  BM25: {result['bm25_score']:.3f}, "
                    f"Semantic: {result['semantic_score']:.3f}"
                )
                print(f"  {result['document'][:100]}...")
        case "rrf-search":
            query = args.query

            if args.enhance == "spell":
                enhanced_query = enhance_query_spell(query)
                print(
                    f"Enhanced query ({args.enhance}): '{query}' -> '{enhanced_query}'\n"
                )
                query = enhanced_query
            elif args.enhance == "rewrite":
                enhanced_query = enhance_query_rewrite(query)
                print(
                    f"Enhanced query ({args.enhance}): '{query}' -> '{enhanced_query}'\n"
                )
                query = enhanced_query
            elif args.enhance == "expand":
                enhanced_query = enhance_query_expand(query)
                print(
                    f"Enhanced query ({args.enhance}): '{query}' -> '{enhanced_query}'\n"
                )
                query = enhanced_query

            documents = load_movies()
            searcher = HybridSearch(documents)
            results = searcher.rrf_search(query, args.k, args.limit)
            if args.rerank_method == "individual":
                print(
                    f"Re-ranking top {len(results)} results using individual method..."
                )
                results = rerank_individual(query, results)
                results = results[: args.limit]
            elif args.rerank_method == "batch":
                print(
                    f"Re-ranking top {len(results)} results using batch method..."
                )
                results = rerank_batch(query, results)[: args.limit]
                print(
                    f"Reciprocal Rank Fusion Results for '{query}' (k={args.k}):\n"
                )
            elif args.rerank_method == "cross_encoder":
                print(
                    f"Reranking top {len(results)} results using cross_encoder method...\n"
                )
                results = rerank_cross_encoder(query, results)[: args.limit]
                print(
                    f"Reciprocal Rank Fusion Results for '{query}' (k={args.k}):\n"
                )
            for i, result in enumerate(results, start=1):
                print(f"{i}. {result['title']}")
                print(f"  RRF Score: {result['rrf_score']:.3f}")
                print(
                    f"  BM25 Rank: {result['bm25_rank']}, "
                    f"Semantic Rank: {result['semantic_rank']}"
                )
                print(f"  {result['document'][:100]}...")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()