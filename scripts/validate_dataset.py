from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES_DIR = PROJECT_ROOT / "data" / "cases"
DEFAULT_GRAPH_PATH = PROJECT_ROOT / "data" / "kinematics_graph.json"

REQUIRED_FIELDS = {
    "case_id",
    "problem_text",
    "known_symbols",
    "unknown",
    "constraints_fired",
    "equations_used",
    "law_nodes",
    "implied_values",
}
VALID_GRADES = {6, 7, 9, 11}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
COMMON_UNITS = {
    "m",
    "cm",
    "km",
    "s",
    "sec",
    "second",
    "seconds",
    "min",
    "mins",
    "minute",
    "minutes",
    "h",
    "hr",
    "hrs",
    "hour",
    "hours",
    "day",
    "days",
    "m/s",
    "m/s^2",
    "km/h",
    "kmph",
    "m/min",
}


@dataclass(frozen=True)
class CaseLocation:
    path: Path
    line: int
    case_id: str = "<unknown>"

    def label(self) -> str:
        relative = self.path.relative_to(PROJECT_ROOT)
        return f"{relative}:{self.line} ({self.case_id})"


@dataclass
class ValidationReport:
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    warnings: list[str] = field(default_factory=list)
    errors_by_case: dict[CaseLocation, list[str]] = field(default_factory=dict)
    duplicate_ids: dict[str, list[CaseLocation]] = field(default_factory=dict)
    duplicate_problem_texts: dict[str, list[CaseLocation]] = field(default_factory=dict)
    invalid_json: list[tuple[CaseLocation, str]] = field(default_factory=list)
    missing_fields: dict[str, list[CaseLocation]] = field(default_factory=lambda: defaultdict(list))
    invalid_equations: dict[str, list[CaseLocation]] = field(default_factory=lambda: defaultdict(list))
    invalid_constraints: dict[str, list[CaseLocation]] = field(default_factory=lambda: defaultdict(list))
    invalid_grades: list[CaseLocation] = field(default_factory=list)
    invalid_unknowns: dict[str, list[CaseLocation]] = field(default_factory=lambda: defaultdict(list))
    missing_final_answers: list[CaseLocation] = field(default_factory=list)
    invalid_units: list[tuple[CaseLocation, str]] = field(default_factory=list)
    empty_trigger_phrases: list[CaseLocation] = field(default_factory=list)
    empty_solution_steps: list[CaseLocation] = field(default_factory=list)
    grade_coverage: Counter = field(default_factory=Counter)
    concept_coverage: Counter = field(default_factory=Counter)
    difficulty_coverage: Counter = field(default_factory=Counter)

    def add_error(self, location: CaseLocation, message: str) -> None:
        self.errors_by_case.setdefault(location, []).append(message)

    @property
    def has_failures(self) -> bool:
        return bool(self.errors_by_case)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate CalcMate JSONL case datasets.")
    parser.add_argument("--cases-dir", type=Path, default=DEFAULT_CASES_DIR)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH_PATH)
    args = parser.parse_args()

    graph_ids = load_graph_ids(args.graph)
    report = validate_cases(args.cases_dir, graph_ids)
    print(render_report(report))
    return 1 if report.has_failures else 0


def load_graph_ids(graph_path: Path) -> dict[str, set[str]]:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    ids_by_type: dict[str, set[str]] = defaultdict(set)
    quantity_symbols: set[str] = set()
    unit_symbols: set[str] = set(COMMON_UNITS)
    for node in graph.get("nodes", []):
        node_id = node.get("id")
        node_type = node.get("type")
        if node_id and node_type:
            ids_by_type[node_type].add(node_id)
        if node_type == "quantity" and node.get("symbol"):
            quantity_symbols.add(str(node["symbol"]))
        if node_type == "unit" and node.get("symbol"):
            unit_symbols.add(str(node["symbol"]))
    ids_by_type["quantity_symbols"] = quantity_symbols
    ids_by_type["unit_symbols"] = unit_symbols
    return ids_by_type


def validate_cases(cases_dir: Path, graph_ids: dict[str, set[str]]) -> ValidationReport:
    report = ValidationReport()
    seen_ids: dict[str, list[CaseLocation]] = defaultdict(list)
    seen_texts: dict[str, list[CaseLocation]] = defaultdict(list)

    for path in sorted(cases_dir.rglob("*.jsonl")):
        for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw_line.strip()
            if not line:
                continue

            report.total_cases += 1
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                location = CaseLocation(path, line_number)
                report.invalid_json.append((location, exc.msg))
                report.add_error(location, f"Invalid JSON: {exc.msg}")
                continue

            case_id = str(data.get("case_id", "<unknown>"))
            location = CaseLocation(path, line_number, case_id)
            validate_case(data, location, graph_ids, report)

            if "case_id" in data:
                seen_ids[case_id].append(location)
            if "problem_text" in data:
                seen_texts[normalize_text(str(data["problem_text"]))].append(location)

    add_duplicate_failures(seen_ids, report.duplicate_ids, report, "Duplicate case_id")
    add_duplicate_failures(seen_texts, report.duplicate_problem_texts, report, "Duplicate problem_text")

    report.failed_cases = len(report.errors_by_case)
    report.passed_cases = report.total_cases - report.failed_cases
    return report


def validate_case(
    data: dict[str, Any],
    location: CaseLocation,
    graph_ids: dict[str, set[str]],
    report: ValidationReport,
) -> None:
    missing = sorted(REQUIRED_FIELDS - set(data))
    for field_name in missing:
        report.missing_fields[field_name].append(location)
        report.add_error(location, f"Missing required field: {field_name}")

    validate_grade(data, location, report)
    validate_difficulty(data, location, report)
    validate_graph_references(data, location, graph_ids, report)
    validate_unknown(data, location, graph_ids, report)
    validate_final_answer(data, location, graph_ids, report)
    validate_non_empty_list(data, "trigger_phrases", report.empty_trigger_phrases, location, report)
    validate_non_empty_list(data, "solution_steps", report.empty_solution_steps, location, report)

    if data.get("grade") is not None:
        report.grade_coverage[str(data["grade"])] += 1
    else:
        report.grade_coverage["<missing>"] += 1
    report.concept_coverage[str(data.get("concept", "<missing>"))] += 1
    report.difficulty_coverage[str(data.get("difficulty", "<missing>"))] += 1


def validate_grade(data: dict[str, Any], location: CaseLocation, report: ValidationReport) -> None:
    grade = data.get("grade")
    if grade is None:
        report.warnings.append(f"{location.label()}: missing grade")
        return
    if grade not in VALID_GRADES:
        report.invalid_grades.append(location)
        report.add_error(location, f"Invalid grade: {grade!r}")


def validate_difficulty(data: dict[str, Any], location: CaseLocation, report: ValidationReport) -> None:
    difficulty = data.get("difficulty")
    if difficulty is None:
        report.warnings.append(f"{location.label()}: missing difficulty")
        return
    if str(difficulty).lower() not in VALID_DIFFICULTIES:
        report.add_error(location, f"Invalid difficulty: {difficulty!r}")


def validate_graph_references(
    data: dict[str, Any],
    location: CaseLocation,
    graph_ids: dict[str, set[str]],
    report: ValidationReport,
) -> None:
    for equation_id in data.get("equations_used", []) or []:
        if equation_id not in graph_ids["equation"]:
            report.invalid_equations[equation_id].append(location)
            report.add_error(location, f"Invalid equation ID: {equation_id}")

    valid_constraints = set(graph_ids["constraint"]) | {"meta_from_rest_u_zero", "meta_free_fall_a_minus_g"}
    for constraint_id in data.get("constraints_fired", []) or []:
        if constraint_id not in valid_constraints:
            report.invalid_constraints[constraint_id].append(location)
            report.add_error(location, f"Invalid constraint ID: {constraint_id}")


def validate_unknown(
    data: dict[str, Any],
    location: CaseLocation,
    graph_ids: dict[str, set[str]],
    report: ValidationReport,
) -> None:
    if "unknown" not in data:
        return
    unknown = str(data["unknown"])
    valid_unknowns = graph_ids["quantity_symbols"]
    if unknown not in valid_unknowns:
        report.invalid_unknowns[unknown].append(location)
        report.add_error(location, f"Invalid unknown variable: {unknown}")


def validate_final_answer(
    data: dict[str, Any],
    location: CaseLocation,
    graph_ids: dict[str, set[str]],
    report: ValidationReport,
) -> None:
    final_answer = data.get("final_answer")
    if final_answer is None:
        report.missing_final_answers.append(location)
        report.add_error(location, "Missing final_answer")
        return
    if not isinstance(final_answer, dict):
        report.add_error(location, "final_answer must be an object")
        return
    if "value" not in final_answer:
        report.add_error(location, "final_answer missing value")
    unit = final_answer.get("unit")
    if not isinstance(unit, str) or not unit.strip():
        report.invalid_units.append((location, "<missing>"))
        report.add_error(location, "final_answer has invalid or missing unit")
        return
    if not is_valid_unit_text(unit, graph_ids["unit_symbols"]):
        report.invalid_units.append((location, unit))
        report.add_error(location, f"Invalid final_answer unit: {unit}")


def validate_non_empty_list(
    data: dict[str, Any],
    field_name: str,
    collection: list[CaseLocation],
    location: CaseLocation,
    report: ValidationReport,
) -> None:
    if field_name not in data:
        collection.append(location)
        report.add_error(location, f"Missing {field_name}")
        return
    value = data[field_name]
    if not isinstance(value, list) or not any(str(item).strip() for item in value):
        collection.append(location)
        report.add_error(location, f"Empty {field_name}")


def is_valid_unit_text(unit: str, valid_units: set[str]) -> bool:
    lowered = unit.strip().lower()
    if lowered.startswith("conceptual"):
        return True
    normalized = lowered.replace("per", "/")
    normalized = normalized.replace("metres", "m").replace("metre", "m")
    normalized = normalized.replace("meters", "m").replace("meter", "m")
    normalized = normalized.replace("seconds", "s").replace("second", "s")
    normalized = normalized.replace("kilometres", "km").replace("kilometre", "km")
    normalized = normalized.replace("kilometers", "km").replace("kilometer", "km")
    normalized = normalized.replace("approx.", "approx")

    unit_pattern = re.compile(r"\b(?:m/s\^2|m/s|km/h|kmph|m/min|cm|km|m|s|min|minutes?|h|hours?|days?)\b")
    tokens = set(unit_pattern.findall(normalized))
    if not tokens:
        return False
    return all(token in valid_units for token in tokens)


def add_duplicate_failures(
    seen: dict[str, list[CaseLocation]],
    target: dict[str, list[CaseLocation]],
    report: ValidationReport,
    message: str,
) -> None:
    for value, locations in seen.items():
        if len(locations) <= 1:
            continue
        target[value] = locations
        for location in locations:
            report.add_error(location, f"{message}: {value}")


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def render_report(report: ValidationReport) -> str:
    lines = [
        "Dataset Validation",
        "",
        f"Total Cases: {report.total_cases}",
        f"Passed: {report.passed_cases}",
        f"Failed: {report.failed_cases}",
        f"Warnings: {len(report.warnings)}",
        "",
        "Duplicate IDs",
        *render_duplicate_section(report.duplicate_ids),
        "",
        "Duplicate Problem Text",
        *render_duplicate_section(report.duplicate_problem_texts),
        "",
        "Invalid JSON",
        *render_invalid_json(report.invalid_json),
        "",
        "Missing Fields",
        *render_grouped_locations(report.missing_fields),
        "",
        "Invalid Equations",
        *render_grouped_locations(report.invalid_equations),
        "",
        "Invalid Constraints",
        *render_grouped_locations(report.invalid_constraints),
        "",
        "Invalid Grades",
        *render_locations(report.invalid_grades),
        "",
        "Invalid Unknowns",
        *render_grouped_locations(report.invalid_unknowns),
        "",
        "Missing Final Answer",
        *render_locations(report.missing_final_answers),
        "",
        "Invalid Units",
        *render_invalid_units(report.invalid_units),
        "",
        "Empty Trigger Phrases",
        *render_locations(report.empty_trigger_phrases),
        "",
        "Empty Solution Steps",
        *render_locations(report.empty_solution_steps),
        "",
        "Coverage by Grade",
        *render_counter(report.grade_coverage),
        "",
        "Coverage by Concept",
        *render_counter(report.concept_coverage),
        "",
        "Coverage by Difficulty",
        *render_counter(report.difficulty_coverage),
    ]
    if report.warnings:
        lines.extend(["", "Warnings Detail", *[f"- {warning}" for warning in report.warnings[:50]]])
        if len(report.warnings) > 50:
            lines.append(f"- ... {len(report.warnings) - 50} more")
    return "\n".join(lines)


def render_duplicate_section(grouped: dict[str, list[CaseLocation]]) -> list[str]:
    if not grouped:
        return ["- None"]
    lines = []
    for value, locations in sorted(grouped.items()):
        labels = ", ".join(location.label() for location in locations)
        lines.append(f"- {value}: {labels}")
    return lines


def render_grouped_locations(grouped: dict[str, list[CaseLocation]]) -> list[str]:
    if not grouped:
        return ["- None"]
    lines = []
    for key, locations in sorted(grouped.items()):
        lines.append(f"- {key}: {len(locations)}")
        for location in locations[:10]:
            lines.append(f"  - {location.label()}")
        if len(locations) > 10:
            lines.append(f"  - ... {len(locations) - 10} more")
    return lines


def render_locations(locations: list[CaseLocation]) -> list[str]:
    if not locations:
        return ["- None"]
    lines = [f"- {location.label()}" for location in locations[:25]]
    if len(locations) > 25:
        lines.append(f"- ... {len(locations) - 25} more")
    return lines


def render_invalid_units(items: list[tuple[CaseLocation, str]]) -> list[str]:
    if not items:
        return ["- None"]
    lines = [f"- {location.label()}: {unit}" for location, unit in items[:25]]
    if len(items) > 25:
        lines.append(f"- ... {len(items) - 25} more")
    return lines


def render_invalid_json(items: list[tuple[CaseLocation, str]]) -> list[str]:
    if not items:
        return ["- None"]
    lines = [f"- {location.label()}: {message}" for location, message in items[:25]]
    if len(items) > 25:
        lines.append(f"- ... {len(items) - 25} more")
    return lines


def render_counter(counter: Counter) -> list[str]:
    if not counter:
        return ["- None"]
    return [f"- {key}: {count}" for key, count in sorted(counter.items(), key=lambda item: str(item[0]))]


if __name__ == "__main__":
    sys.exit(main())
