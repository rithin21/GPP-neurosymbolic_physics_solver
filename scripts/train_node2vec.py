from __future__ import annotations

"""Build the case-similarity graph and train node2vec structural embeddings.

Run:
    python scripts/train_node2vec.py
"""

import argparse

from calcmate.case_graph import (
    DEFAULT_EMBEDDINGS_PATH,
    build_case_graph,
    generate_walks,
    load_all_cases,
    save_embeddings,
    train_skipgram,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train node2vec embeddings over the kinematics case graph.")
    parser.add_argument("--num-walks", type=int, default=10)
    parser.add_argument("--walk-length", type=int, default=20)
    parser.add_argument("--p", type=float, default=1.0, help="Return parameter (higher = less backtracking).")
    parser.add_argument("--q", type=float, default=2.0, help="In-out parameter (higher = more local exploration).")
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output", type=str, default=str(DEFAULT_EMBEDDINGS_PATH))
    args = parser.parse_args()

    cases = load_all_cases()
    print(f"Loaded {len(cases)} cases.")

    graph = build_case_graph(cases)
    print(f"Case graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")

    walks = generate_walks(graph, num_walks=args.num_walks, walk_length=args.walk_length, p=args.p, q=args.q)
    print(f"Generated {len(walks)} random walks.")

    embeddings = train_skipgram(walks, dim=args.dim, window=args.window, epochs=args.epochs)
    print(f"Trained embeddings for {len(embeddings)} cases.")

    save_embeddings(embeddings, args.output)
    print(f"Saved embeddings to {args.output}")


if __name__ == "__main__":
    main()
