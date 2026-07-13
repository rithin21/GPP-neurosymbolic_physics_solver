from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calcmate.retrieval import FaissCaseRetriever  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the FAISS index for CalcMate solved cases.")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("data/cases/kinematics"),
        help="Path to a solved-cases JSONL file or a chapter directory of them.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("data/cases/kinematics_cases.faiss"),
        help="Output FAISS index path.",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=Path("data/cases/kinematics_cases.meta.json"),
        help="Output metadata JSON path.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name.",
    )
    args = parser.parse_args()

    retriever = FaissCaseRetriever(
        cases_path=args.cases,
        index_path=args.index,
        meta_path=args.meta,
        embedding_model=args.model,
    )

    print("Cases source:", args.cases)
    print("Total cases loaded:", len(retriever.cases))
    for case in retriever.cases[:5]:
        print(" ", case.case_id)

    retriever.build_index()
    print(f"Built FAISS index at {args.index}")
    print(f"Wrote metadata at {args.meta}")


if __name__ == "__main__":
    main()

