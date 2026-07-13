from __future__ import annotations

"""Replay-validate dataset cases through the reasoning-program VM.

For every case it executes the ``reasoning_program`` and checks it reproduces
``final_answer``. Also runs light structural checks (canonical target symbol,
trigger phrases present in the text) and prints coverage by grade/concept.

Usage:
    python scripts/replay_validate.py [file1.jsonl file2.jsonl ...]

With no args it scans data/kinematics_cases_*.jsonl.
"""

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from calcmate.program_executor import ProgramExecutor  # noqa: E402

CANONICAL_SYMBOLS = {"u", "v", "a", "t", "s"}


def iter_cases(paths):
    for path in paths:
        for line_no, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield path, line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                yield path, line_no, {"__json_error__": str(exc)}


def main() -> int:
    args = sys.argv[1:]
    if args:
        paths = [Path(a) for a in args]
    else:
        paths = sorted((PROJECT_ROOT / "data").glob("kinematics_cases_*.jsonl"))
    if not paths:
        print("No dataset files found.")
        return 1

    executor = ProgramExecutor()
    total = passed = 0
    replay_fail = []
    warnings = []
    grade_cov = Counter()
    concept_cov = Counter()
    difficulty_cov = Counter()

    for path, line_no, case in iter_cases(paths):
        total += 1
        label = f"{Path(path).name}:{line_no} ({case.get('case_id', '?')})"

        if "__json_error__" in case:
            replay_fail.append((label, f"invalid JSON: {case['__json_error__']}"))
            continue

        grade_cov[str(case.get("grade", "?"))] += 1
        concept_cov[str(case.get("concept", "?"))] += 1
        difficulty_cov[str(case.get("difficulty", "?"))] += 1

        # Structural warnings (non-fatal)
        tsym = case.get("target", {}).get("symbol")
        if tsym not in CANONICAL_SYMBOLS:
            warnings.append(f"{label}: non-canonical target symbol {tsym!r}")
        text = (case.get("problem_text") or "").lower()
        for phrase in case.get("trigger_phrases", []):
            if str(phrase).lower() not in text:
                warnings.append(f"{label}: trigger phrase {phrase!r} not in problem_text")

        # The real check: replay the program
        result = executor.execute(case)
        if result.ok:
            passed += 1
        else:
            replay_fail.append((label, result.error))

    print("=" * 70)
    print(f"REPLAY VALIDATION  |  files: {len(paths)}  cases: {total}")
    print(f"  PASSED: {passed}    FAILED: {len(replay_fail)}    WARNINGS: {len(warnings)}")
    print("=" * 70)

    if replay_fail:
        print("\nREPLAY FAILURES (program did not reproduce final_answer):")
        for label, reason in replay_fail[:40]:
            print(f"  - {label}: {reason}")
        if len(replay_fail) > 40:
            print(f"  ... {len(replay_fail) - 40} more")

    if warnings:
        print("\nWARNINGS:")
        for w in warnings[:25]:
            print(f"  - {w}")
        if len(warnings) > 25:
            print(f"  ... {len(warnings) - 25} more")

    print("\nCOVERAGE by grade:", dict(sorted(grade_cov.items())))
    print("COVERAGE by concept:", dict(concept_cov))
    print("COVERAGE by difficulty:", dict(difficulty_cov))

    return 1 if replay_fail else 0


if __name__ == "__main__":
    sys.exit(main())
