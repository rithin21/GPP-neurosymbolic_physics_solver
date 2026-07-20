from __future__ import annotations

"""Generate a verified dataset of ~200 kinematics solved cases.

Every generated case is solved with SymPy against the *actual* equation
expression stored in data/kinematics_graph.json, so the recorded
final_answer is guaranteed consistent with what calcmate.reasoning would
produce - unlike the older data/cases/kinematics_cases.jsonl records, some
of which reference equation/constraint IDs that don't exist in the graph.

Schema per line (aligned with scripts/validate_dataset.py REQUIRED_FIELDS):
  case_id, chapter, grade, concept, subconcept, difficulty, problem_text,
  known_symbols (plain symbols, e.g. ["u","a","t"]), derived_symbols,
  unknown, trigger_phrases, constraints_fired, equations_used, law_nodes,
  implied_values, method, solution_steps, final_answer{value,unit}.

Run:
    python scripts/generate_kinematics_cases.py
"""

import json
import random
from pathlib import Path
from typing import Callable

import sympy as sp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "cases" / "kinematics_suvat_cases.jsonl"

UNIT_BY_SYMBOL = {
    "u": "m/s", "v": "m/s", "a": "m/s^2", "t": "s", "s": "m",
    "speed": "m/s", "distance": "m", "time": "s",
    "avg_v": "m/s", "v1": "m/s", "v2": "m/s", "relative_speed": "m/s",
    "direction": "", "separation": "m",
}

_SYMS = {
    name: sp.Symbol(name)
    for name in (
        "u", "v", "a", "t", "s", "speed", "distance", "time",
        "avg_v", "v1", "v2", "relative_speed", "direction", "separation",
    )
}

random.seed(20260717)


def _solve(expression: str, knowns: dict[str, float], target: str) -> float:
    left, right = expression.split("=")
    equation = sp.Eq(sp.sympify(left.strip(), locals=_SYMS), sp.sympify(right.strip(), locals=_SYMS))
    substitutions = {_SYMS[symbol]: value for symbol, value in knowns.items()}
    solved = sp.solve(equation.subs(substitutions), _SYMS[target])
    numeric = [float(root) for root in solved if root.is_number]
    if not numeric:
        raise ValueError(f"No numeric solution for {target} in {expression} with {knowns}")
    if target == "v" and len(numeric) > 1:
        positive = [value for value in numeric if value >= 0]
        return round(positive[0] if positive else numeric[0], 4)
    return round(numeric[0], 4)


def _case(
    case_id: str,
    *,
    grade: int,
    concept: str,
    subconcept: str,
    difficulty: str,
    problem_text: str,
    equation_id: str,
    expression: str,
    law_node: str,
    knowns: dict[str, float],
    target: str,
    trigger_phrases: list[str],
    constraints_fired: list[str],
    implied_values: dict[str, float],
    method: str,
    solution_steps: list[str],
) -> dict:
    all_knowns = {**knowns, **implied_values}
    value = _solve(expression, all_knowns, target)
    return {
        "case_id": case_id,
        "chapter": "kinematics",
        "grade": grade,
        "concept": concept,
        "subconcept": subconcept,
        "difficulty": difficulty,
        "problem_text": problem_text,
        "known_symbols": sorted(knowns),
        "derived_symbols": sorted(implied_values),
        "unknown": target,
        "trigger_phrases": trigger_phrases,
        "constraints_fired": constraints_fired,
        "equations_used": [equation_id],
        "law_nodes": [law_node],
        "implied_values": implied_values,
        "method": method,
        "solution_steps": solution_steps,
        "final_answer": {"value": value, "unit": UNIT_BY_SYMBOL[target]},
    }


def _rand(lo: float, hi: float, step: float = 0.5) -> float:
    steps = int((hi - lo) / step)
    return round(lo + step * random.randint(0, steps), 2)


def _rand_int(lo: int, hi: int) -> int:
    return random.randint(lo, hi)


def _append_unique(cases: list[dict], seen_texts: set[str], build: Callable[[], dict]) -> None:
    """Call `build()` (which draws fresh random values each attempt) until
    it produces a case whose problem_text hasn't been used yet."""
    for _ in range(100):
        case = build()
        normalized = " ".join(case["problem_text"].lower().split())
        if normalized in seen_texts:
            continue
        seen_texts.add(normalized)
        cases.append(case)
        return
    raise RuntimeError(f"Could not generate a unique problem_text for {build}")


LAW_UNIFORM = "law_uniform_motion"
LAW_ACCEL = "law_kinematics_uniform_acceleration"
LAW_RELATIVE = "law_relative_motion"


def generate_eq_v_u_at(cases: list[dict], seen: set[str]) -> None:
    """v = u + a*t - first equation of motion. All four targets are linear
    (single root), so every permutation is safe to solve."""
    subjects = ["car", "motorbike", "cyclist", "train", "scooter", "bus", "runner", "skater"]

    for i in range(12):
        subject = subjects[i % len(subjects)]

        def build(subject=subject, i=i):
            u = _rand(0, 25, 1)
            a = _rand(0.5, 4, 0.5)
            t = _rand_int(2, 12)
            return _case(
                f"suvat_eq1_find_v_{i:03d}", grade=9, concept="Uniform Acceleration (SUVAT)",
                subconcept="First Equation of Motion", difficulty="easy",
                problem_text=(
                    f"A {subject} moving at {u:g} m/s accelerates uniformly at {a:g} m/s^2 "
                    f"for {t:g} s. Find the final velocity."
                ),
                equation_id="eq_v_u_at", expression="v = u + a*t", law_node=LAW_ACCEL,
                knowns={"u": u, "a": a, "t": t}, target="v", trigger_phrases=["accelerates uniformly"],
                constraints_fired=[], implied_values={},
                method="Direct substitution into the first equation of motion (v = u + at), solved for v.",
                solution_steps=[
                    "Identify knowns u, a, t.",
                    "Substitute into v = u + a*t.",
                    "Compute the numeric result.",
                ],
            )

        _append_unique(cases, seen, build)

    for i in range(12):
        subject = subjects[(i + 3) % len(subjects)]

        def build(subject=subject, i=i):
            v = _rand(10, 40, 1)
            a = _rand(0.5, 4, 0.5)
            t = _rand_int(2, 12)
            return _case(
                f"suvat_eq1_find_u_{i:03d}", grade=9, concept="Uniform Acceleration (SUVAT)",
                subconcept="First Equation of Motion", difficulty="medium",
                problem_text=(
                    f"A {subject} reaches a velocity of {v:g} m/s after accelerating at {a:g} m/s^2 "
                    f"for {t:g} s. Find its initial velocity."
                ),
                equation_id="eq_v_u_at", expression="v = u + a*t", law_node=LAW_ACCEL,
                knowns={"v": v, "a": a, "t": t}, target="u", trigger_phrases=["accelerating"],
                constraints_fired=[], implied_values={},
                method="Rearrange the first equation of motion (v = u + at) to solve for u.",
                solution_steps=[
                    "Identify knowns v, a, t.",
                    "Rearrange v = u + a*t to u = v - a*t.",
                    "Compute the numeric result.",
                ],
            )

        _append_unique(cases, seen, build)

    for i in range(12):
        subject = subjects[(i + 5) % len(subjects)]

        def build(subject=subject, i=i):
            u = _rand(0, 20, 1)
            v = _rand(20, 45, 1)
            t = _rand_int(3, 10)
            return _case(
                f"suvat_eq1_find_a_{i:03d}", grade=9, concept="Uniform Acceleration (SUVAT)",
                subconcept="First Equation of Motion", difficulty="medium",
                problem_text=(
                    f"A {subject}'s velocity increases from {u:g} m/s to {v:g} m/s in {t:g} s "
                    f"under uniform acceleration. Find the acceleration."
                ),
                equation_id="eq_v_u_at", expression="v = u + a*t", law_node=LAW_ACCEL,
                knowns={"u": u, "v": v, "t": t}, target="a", trigger_phrases=["uniform acceleration"],
                constraints_fired=[], implied_values={},
                method="Rearrange the first equation of motion (v = u + at) to solve for a.",
                solution_steps=[
                    "Identify knowns u, v, t.",
                    "Rearrange v = u + a*t to a = (v - u)/t.",
                    "Compute the numeric result.",
                ],
            )

        _append_unique(cases, seen, build)

    for i in range(12):
        subject = subjects[(i + 7) % len(subjects)]

        def build(subject=subject, i=i):
            u = _rand(0, 20, 1)
            v = _rand(20, 45, 1)
            a = _rand(1, 5, 0.5)
            return _case(
                f"suvat_eq1_find_t_{i:03d}", grade=9, concept="Uniform Acceleration (SUVAT)",
                subconcept="First Equation of Motion", difficulty="medium",
                problem_text=(
                    f"A {subject} speeds up from {u:g} m/s to {v:g} m/s with a uniform acceleration "
                    f"of {a:g} m/s^2. Find the time taken."
                ),
                equation_id="eq_v_u_at", expression="v = u + a*t", law_node=LAW_ACCEL,
                knowns={"u": u, "v": v, "a": a}, target="t", trigger_phrases=["uniform acceleration"],
                constraints_fired=[], implied_values={},
                method="Rearrange the first equation of motion (v = u + at) to solve for t.",
                solution_steps=[
                    "Identify knowns u, v, a.",
                    "Rearrange v = u + a*t to t = (v - u)/a.",
                    "Compute the numeric result.",
                ],
            )

        _append_unique(cases, seen, build)


def generate_eq_s_vt(cases: list[dict], seen: set[str]) -> None:
    """s = v*t - uniform motion, no acceleration. All three targets safe."""
    subjects = ["ferry", "cargo train", "conveyor belt", "elevator", "toy car", "drone"]

    for i in range(10):
        subject = subjects[i % len(subjects)]

        def build(subject=subject, i=i):
            v = _rand(2, 20, 0.5)
            t = _rand_int(2, 20)
            return _case(
                f"suvat_eqs_find_s_{i:03d}", grade=9, concept="Uniform Motion (SUVAT)",
                subconcept="Distance under Uniform Motion", difficulty="easy",
                problem_text=(
                    f"A {subject} moves at a constant speed of {v:g} m/s for {t:g} s. "
                    f"Find the distance covered."
                ),
                equation_id="eq_s_vt", expression="s = v*t", law_node=LAW_UNIFORM,
                knowns={"v": v, "t": t}, target="s", trigger_phrases=["constant speed"],
                constraints_fired=["constraint_constant_speed"], implied_values={},
                method="Constant speed implies zero acceleration; apply s = v*t directly.",
                solution_steps=[
                    "Recognize constant speed means a = 0.",
                    "Substitute v and t into s = v*t.",
                    "Compute the numeric result.",
                ],
            )

        _append_unique(cases, seen, build)

    for i in range(10):
        subject = subjects[(i + 2) % len(subjects)]

        def build(subject=subject, i=i):
            s = _rand_int(20, 400)
            t = _rand_int(4, 25)
            return _case(
                f"suvat_eqs_find_v_{i:03d}", grade=9, concept="Uniform Motion (SUVAT)",
                subconcept="Speed under Uniform Motion", difficulty="easy",
                problem_text=(
                    f"A {subject} covers {s:g} m in {t:g} s while moving at constant speed. "
                    f"Find its speed."
                ),
                equation_id="eq_s_vt", expression="s = v*t", law_node=LAW_UNIFORM,
                knowns={"s": s, "t": t}, target="v", trigger_phrases=["constant speed", "covers"],
                constraints_fired=["constraint_constant_speed"], implied_values={},
                method="Constant speed implies zero acceleration; rearrange s = v*t to v = s/t.",
                solution_steps=[
                    "Recognize constant speed means a = 0.",
                    "Rearrange s = v*t to v = s/t.",
                    "Compute the numeric result.",
                ],
            )

        _append_unique(cases, seen, build)

    for i in range(10):
        subject = subjects[(i + 4) % len(subjects)]

        def build(subject=subject, i=i):
            s = _rand_int(20, 400)
            v = _rand(2, 20, 0.5)
            return _case(
                f"suvat_eqs_find_t_{i:03d}", grade=9, concept="Uniform Motion (SUVAT)",
                subconcept="Time under Uniform Motion", difficulty="easy",
                problem_text=(
                    f"A {subject} travels {s:g} m at a uniform speed of {v:g} m/s. "
                    f"Find the time taken."
                ),
                equation_id="eq_s_vt", expression="s = v*t", law_node=LAW_UNIFORM,
                knowns={"s": s, "v": v}, target="t", trigger_phrases=["uniform motion", "travels"],
                constraints_fired=["constraint_constant_speed"], implied_values={},
                method="Constant speed implies zero acceleration; rearrange s = v*t to t = s/v.",
                solution_steps=[
                    "Recognize constant speed means a = 0.",
                    "Rearrange s = v*t to t = s/v.",
                    "Compute the numeric result.",
                ],
            )

        _append_unique(cases, seen, build)


def generate_eq_average_speed(cases: list[dict], seen: set[str]) -> None:
    """speed = distance/time - grade 6/7 style plain-quantity problems."""
    subjects = ["hiker", "delivery van", "swimmer", "school bus", "rickshaw", "jogger", "tortoise"]

    for i in range(10):
        subject = subjects[i % len(subjects)]

        def build(subject=subject, i=i):
            distance = _rand_int(10, 500)
            time = _rand_int(5, 60)
            return _case(
                f"grade7_avgspeed_find_speed_{i:03d}", grade=7, concept="Uniform Motion",
                subconcept="Average Speed", difficulty="easy",
                problem_text=f"A {subject} covers {distance:g} m in {time:g} s. Find its speed.",
                equation_id="eq_average_speed", expression="speed = distance/time", law_node=LAW_UNIFORM,
                knowns={"distance": distance, "time": time}, target="speed", trigger_phrases=["covers"],
                constraints_fired=[], implied_values={},
                method="Apply speed = distance / time directly.",
                solution_steps=["Identify distance and time.", "Divide distance by time.", "State the numeric speed."],
            )

        _append_unique(cases, seen, build)

    for i in range(10):
        subject = subjects[(i + 2) % len(subjects)]

        def build(subject=subject, i=i):
            speed = _rand(0.5, 20, 0.5)
            time = _rand_int(5, 60)
            return _case(
                f"grade7_avgspeed_find_distance_{i:03d}", grade=7, concept="Uniform Motion",
                subconcept="Distance from Speed and Time", difficulty="easy",
                problem_text=f"A {subject} moves at {speed:g} m/s for {time:g} s. Find the distance travelled.",
                equation_id="eq_average_speed", expression="speed = distance/time", law_node=LAW_UNIFORM,
                knowns={"speed": speed, "time": time}, target="distance", trigger_phrases=["travelled", "moves"],
                constraints_fired=[], implied_values={},
                method="Rearrange speed = distance / time to distance = speed * time.",
                solution_steps=["Identify speed and time.", "Multiply speed by time.", "State the numeric distance."],
            )

        _append_unique(cases, seen, build)

    for i in range(10):
        subject = subjects[(i + 4) % len(subjects)]

        def build(subject=subject, i=i):
            distance = _rand_int(10, 500)
            speed = _rand(0.5, 20, 0.5)
            return _case(
                f"grade7_avgspeed_find_time_{i:03d}", grade=7, concept="Uniform Motion",
                subconcept="Time from Speed and Distance", difficulty="easy",
                problem_text=(
                    f"A {subject} needs to cover {distance:g} m at a steady speed of {speed:g} m/s. "
                    f"Find the time required."
                ),
                equation_id="eq_average_speed", expression="speed = distance/time", law_node=LAW_UNIFORM,
                knowns={"distance": distance, "speed": speed}, target="time", trigger_phrases=["steady speed"],
                constraints_fired=[], implied_values={},
                method="Rearrange speed = distance / time to time = distance / speed.",
                solution_steps=["Identify distance and speed.", "Divide distance by speed.", "State the numeric time."],
            )

        _append_unique(cases, seen, build)


def generate_eq_s_ut_half_at2(cases: list[dict], seen: set[str]) -> None:
    """s = u*t + a*t^2/2, solved for s only (linear, no root ambiguity)."""
    subjects = ["rocket sled", "roller coaster car", "sprinter", "toboggan", "electric car"]

    for i in range(15):
        subject = subjects[i % len(subjects)]

        def build(subject=subject, i=i):
            u = _rand(0, 15, 1)
            a = _rand(0.5, 5, 0.5)
            t = _rand_int(2, 10)
            return _case(
                f"suvat_eq2_find_s_{i:03d}", grade=9, concept="Uniform Acceleration (SUVAT)",
                subconcept="Second Equation of Motion", difficulty="medium",
                problem_text=(
                    f"A {subject} starts at {u:g} m/s and accelerates at {a:g} m/s^2 for {t:g} s. "
                    f"Find the distance covered."
                ),
                equation_id="eq_s_ut_half_at2", expression="s = u*t + a*t**2/2", law_node=LAW_ACCEL,
                knowns={"u": u, "a": a, "t": t}, target="s", trigger_phrases=["accelerates"],
                constraints_fired=[], implied_values={},
                method="Direct substitution into the second equation of motion (s = ut + 1/2 at^2), solved for s.",
                solution_steps=[
                    "Identify knowns u, a, t.",
                    "Substitute into s = u*t + a*t^2/2.",
                    "Compute the numeric result.",
                ],
            )

        _append_unique(cases, seen, build)


def generate_eq_v2_u2_2as(cases: list[dict], seen: set[str]) -> None:
    """v^2 = u^2 + 2as, targets v/s/a only (target=u is quadratic, excluded
    to avoid the engine's positive-root heuristic picking the wrong branch)."""
    subjects = ["sprinter", "race car", "aircraft", "cyclist", "elevator", "speedboat"]

    for i in range(10):
        subject = subjects[i % len(subjects)]

        def build(subject=subject, i=i):
            u = _rand(0, 15, 1)
            a = _rand(0.5, 5, 0.5)
            s = _rand_int(10, 150)
            return _case(
                f"suvat_eq3_find_v_{i:03d}", grade=9, concept="Uniform Acceleration (SUVAT)",
                subconcept="Third Equation of Motion", difficulty="medium",
                problem_text=(
                    f"A {subject} starts at {u:g} m/s and accelerates at {a:g} m/s^2 over a distance "
                    f"of {s:g} m. Find the final velocity."
                ),
                equation_id="eq_v2_u2_2as", expression="v**2 = u**2 + 2*a*s", law_node=LAW_ACCEL,
                knowns={"u": u, "a": a, "s": s}, target="v", trigger_phrases=["accelerates"],
                constraints_fired=[], implied_values={},
                method=(
                    "Direct substitution into the third equation of motion (v^2 = u^2 + 2as), "
                    "solved for v (positive root)."
                ),
                solution_steps=[
                    "Identify knowns u, a, s.",
                    "Substitute into v^2 = u^2 + 2*a*s.",
                    "Take the positive square root.",
                ],
            )

        _append_unique(cases, seen, build)

    for i in range(10):
        subject = subjects[(i + 2) % len(subjects)]

        def build(subject=subject, i=i):
            u = _rand(0, 15, 1)
            v = _rand(15, 35, 1)
            a = _rand(0.5, 5, 0.5)
            return _case(
                f"suvat_eq3_find_s_{i:03d}", grade=9, concept="Uniform Acceleration (SUVAT)",
                subconcept="Third Equation of Motion", difficulty="medium",
                problem_text=(
                    f"A {subject} accelerates at {a:g} m/s^2 from {u:g} m/s to {v:g} m/s. "
                    f"Find the distance covered."
                ),
                equation_id="eq_v2_u2_2as", expression="v**2 = u**2 + 2*a*s", law_node=LAW_ACCEL,
                knowns={"u": u, "v": v, "a": a}, target="s", trigger_phrases=["accelerates"],
                constraints_fired=[], implied_values={},
                method="Rearrange the third equation of motion (v^2 = u^2 + 2as) to solve for s.",
                solution_steps=[
                    "Identify knowns u, v, a.",
                    "Rearrange v^2 = u^2 + 2*a*s to s = (v^2 - u^2) / (2*a).",
                    "Compute the numeric result.",
                ],
            )

        _append_unique(cases, seen, build)

    for i in range(10):
        subject = subjects[(i + 4) % len(subjects)]

        def build(subject=subject, i=i):
            u = _rand(0, 15, 1)
            v = _rand(15, 35, 1)
            s = _rand_int(10, 150)
            return _case(
                f"suvat_eq3_find_a_{i:03d}", grade=9, concept="Uniform Acceleration (SUVAT)",
                subconcept="Third Equation of Motion", difficulty="hard",
                problem_text=(
                    f"A {subject} increases its speed from {u:g} m/s to {v:g} m/s over {s:g} m. "
                    f"Find its acceleration."
                ),
                equation_id="eq_v2_u2_2as", expression="v**2 = u**2 + 2*a*s", law_node=LAW_ACCEL,
                knowns={"u": u, "v": v, "s": s}, target="a", trigger_phrases=["increases its speed"],
                constraints_fired=[], implied_values={},
                method="Rearrange the third equation of motion (v^2 = u^2 + 2as) to solve for a.",
                solution_steps=[
                    "Identify knowns u, v, s.",
                    "Rearrange v^2 = u^2 + 2*a*s to a = (v^2 - u^2) / (2*s).",
                    "Compute the numeric result.",
                ],
            )

        _append_unique(cases, seen, build)


def generate_from_rest_cases(cases: list[dict], seen: set[str]) -> None:
    subjects = ["car", "cyclist", "sprinter", "train", "elevator"]

    for i in range(8):
        subject = subjects[i % len(subjects)]

        def build(subject=subject, i=i):
            a = _rand(0.5, 4, 0.5)
            t = _rand_int(3, 12)
            return _case(
                f"suvat_from_rest_find_v_{i:03d}", grade=9, concept="Uniform Acceleration (SUVAT)",
                subconcept="Starting from Rest", difficulty="medium",
                problem_text=(
                    f"A {subject} starts from rest and accelerates at {a:g} m/s^2 for {t:g} s. "
                    f"Find its final velocity."
                ),
                equation_id="eq_v_u_at", expression="v = u + a*t", law_node=LAW_ACCEL,
                knowns={"a": a, "t": t}, target="v", trigger_phrases=["starts from rest"],
                constraints_fired=["constraint_from_rest"], implied_values={"u": 0.0},
                method="Apply the from-rest constraint (u = 0), then use the first equation of motion (v = u + at).",
                solution_steps=[
                    "Starting from rest implies u = 0.",
                    "Substitute u = 0 into v = u + a*t.",
                    "Compute the numeric result.",
                ],
            )

        _append_unique(cases, seen, build)

    for i in range(8):
        subject = subjects[(i + 1) % len(subjects)]

        def build(subject=subject, i=i):
            a = _rand(0.5, 4, 0.5)
            t = _rand_int(3, 12)
            return _case(
                f"suvat_from_rest_find_s_{i:03d}", grade=9, concept="Uniform Acceleration (SUVAT)",
                subconcept="Starting from Rest", difficulty="medium",
                problem_text=(
                    f"A {subject} starts from rest and accelerates at {a:g} m/s^2 for {t:g} s. "
                    f"Find the distance it covers."
                ),
                equation_id="eq_s_ut_half_at2", expression="s = u*t + a*t**2/2", law_node=LAW_ACCEL,
                knowns={"a": a, "t": t}, target="s", trigger_phrases=["starts from rest"],
                constraints_fired=["constraint_from_rest"], implied_values={"u": 0.0},
                method=(
                    "Apply the from-rest constraint (u = 0), then use the second equation of "
                    "motion (s = ut + 1/2 at^2)."
                ),
                solution_steps=[
                    "Starting from rest implies u = 0.",
                    "Substitute u = 0 into s = u*t + a*t^2/2.",
                    "Compute the numeric result.",
                ],
            )

        _append_unique(cases, seen, build)


def generate_comes_to_rest_cases(cases: list[dict], seen: set[str]) -> None:
    subjects = ["car", "cyclist", "train", "skateboarder", "bus"]

    for i in range(8):
        subject = subjects[i % len(subjects)]

        def build(subject=subject, i=i):
            u = _rand(10, 30, 1)
            a = -_rand(0.5, 4, 0.5)
            return _case(
                f"suvat_comes_to_rest_find_t_{i:03d}", grade=9, concept="Uniform Acceleration (SUVAT)",
                subconcept="Coming to Rest", difficulty="medium",
                problem_text=(
                    f"A {subject} moving at {u:g} m/s decelerates at {abs(a):g} m/s^2 and comes to rest. "
                    f"Find the time taken to stop."
                ),
                equation_id="eq_v_u_at", expression="v = u + a*t", law_node=LAW_ACCEL,
                knowns={"u": u, "a": a}, target="t", trigger_phrases=["comes to rest"],
                constraints_fired=["constraint_comes_to_rest"], implied_values={"v": 0.0},
                method="Apply the comes-to-rest constraint (v = 0), then use the first equation of motion (v = u + at).",
                solution_steps=[
                    "Coming to rest implies v = 0.",
                    "Rearrange v = u + a*t to t = -u/a.",
                    "Compute the numeric result.",
                ],
            )

        _append_unique(cases, seen, build)

    for i in range(8):
        subject = subjects[(i + 1) % len(subjects)]

        def build(subject=subject, i=i):
            u = _rand(10, 30, 1)
            a = -_rand(0.5, 4, 0.5)
            return _case(
                f"suvat_comes_to_rest_find_s_{i:03d}", grade=9, concept="Uniform Acceleration (SUVAT)",
                subconcept="Coming to Rest", difficulty="hard",
                problem_text=(
                    f"A {subject} moving at {u:g} m/s decelerates uniformly at {abs(a):g} m/s^2 until it "
                    f"comes to rest. Find the stopping distance."
                ),
                equation_id="eq_v2_u2_2as", expression="v**2 = u**2 + 2*a*s", law_node=LAW_ACCEL,
                knowns={"u": u, "a": a}, target="s", trigger_phrases=["comes to rest"],
                constraints_fired=["constraint_comes_to_rest"], implied_values={"v": 0.0},
                method=(
                    "Apply the comes-to-rest constraint (v = 0), then use the third equation of "
                    "motion (v^2 = u^2 + 2as)."
                ),
                solution_steps=[
                    "Coming to rest implies v = 0.",
                    "Rearrange v^2 = u^2 + 2*a*s to s = -u^2 / (2*a).",
                    "Compute the numeric result.",
                ],
            )

        _append_unique(cases, seen, build)


def generate_max_height_cases(cases: list[dict], seen: set[str]) -> None:
    for i in range(12):

        def build(i=i):
            u = _rand(5, 30, 1)
            return _case(
                f"suvat_max_height_{i:03d}", grade=11, concept="Free Fall Under Gravity",
                subconcept="Maximum Height", difficulty="hard",
                problem_text=(
                    f"A ball is thrown upward with an initial velocity of {u:g} m/s. "
                    f"Find the maximum height it reaches."
                ),
                equation_id="eq_v2_u2_2as", expression="v**2 = u**2 + 2*a*s", law_node=LAW_ACCEL,
                knowns={"u": u}, target="s",
                trigger_phrases=["thrown upward", "maximum height"],
                constraints_fired=["constraint_max_height_v_zero", "constraint_free_fall_upward_a_minus_g"],
                implied_values={"v": 0.0, "a": -9.8},
                method=(
                    "Apply the zero-velocity-at-maximum-height constraint (v = 0) and the upward "
                    "free-fall constraint (a = -9.8 m/s^2), then use the third equation of motion "
                    "(v^2 = u^2 + 2as) to solve for s."
                ),
                solution_steps=[
                    "Maximum height implies v = 0.",
                    "Upward free fall implies a = -9.8 m/s^2.",
                    "Substitute into v^2 = u^2 + 2*a*s and solve for s.",
                ],
            )

        _append_unique(cases, seen, build)


def generate_eq2_additional_targets(cases: list[dict], seen: set[str]) -> None:
    """s = u*t + a*t^2/2 - also solve for u or a (both linear given the
    other two knowns; only target=t is quadratic and stays excluded)."""
    subjects = ["rocket sled", "roller coaster car", "sprinter", "toboggan", "electric car", "jet ski"]

    for i in range(12):
        subject = subjects[i % len(subjects)]

        def build(subject=subject, i=i):
            u = _rand(0, 15, 1)
            a = _rand(0.5, 5, 0.5)
            t = _rand_int(2, 10)
            s = round(u * t + 0.5 * a * t**2, 4)
            return _case(
                f"suvat_eq2_find_u_{i:03d}", grade=9, concept="Uniform Acceleration (SUVAT)",
                subconcept="Second Equation of Motion", difficulty="hard",
                problem_text=(
                    f"A {subject} covers {s:g} m in {t:g} s while accelerating at {a:g} m/s^2. "
                    f"Find its initial velocity."
                ),
                equation_id="eq_s_ut_half_at2", expression="s = u*t + a*t**2/2", law_node=LAW_ACCEL,
                knowns={"s": s, "a": a, "t": t}, target="u", trigger_phrases=["accelerating"],
                constraints_fired=[], implied_values={},
                method="Rearrange the second equation of motion (s = ut + 1/2 at^2) to solve for u.",
                solution_steps=[
                    "Identify knowns s, a, t.",
                    "Rearrange s = u*t + a*t^2/2 to u = (s - a*t^2/2) / t.",
                    "Compute the numeric result.",
                ],
            )

        _append_unique(cases, seen, build)

    for i in range(12):
        subject = subjects[(i + 2) % len(subjects)]

        def build(subject=subject, i=i):
            u = _rand(0, 15, 1)
            a = _rand(0.5, 5, 0.5)
            t = _rand_int(2, 10)
            s = round(u * t + 0.5 * a * t**2, 4)
            return _case(
                f"suvat_eq2_find_a_{i:03d}", grade=9, concept="Uniform Acceleration (SUVAT)",
                subconcept="Second Equation of Motion", difficulty="hard",
                problem_text=(
                    f"A {subject} starts at {u:g} m/s and covers {s:g} m in {t:g} s. "
                    f"Find its acceleration."
                ),
                equation_id="eq_s_ut_half_at2", expression="s = u*t + a*t**2/2", law_node=LAW_ACCEL,
                knowns={"s": s, "u": u, "t": t}, target="a", trigger_phrases=["starts at"],
                constraints_fired=[], implied_values={},
                method="Rearrange the second equation of motion (s = ut + 1/2 at^2) to solve for a.",
                solution_steps=[
                    "Identify knowns s, u, t.",
                    "Rearrange s = u*t + a*t^2/2 to a = 2*(s - u*t) / t^2.",
                    "Compute the numeric result.",
                ],
            )

        _append_unique(cases, seen, build)


def generate_free_fall_dropped_cases(cases: list[dict], seen: set[str]) -> None:
    """Dropped from rest: u = 0 (from-rest) and a = -9.8 (free fall) both
    apply, then eq1 gives v and eq2 gives s (both linear substitutions)."""
    subjects = ["ball", "stone", "coin", "apple", "keychain", "wrench"]

    for i in range(10):
        subject = subjects[i % len(subjects)]

        def build(subject=subject, i=i):
            t = _rand_int(1, 8)
            return _case(
                f"suvat_free_fall_dropped_find_v_{i:03d}", grade=11, concept="Free Fall Under Gravity",
                subconcept="Free Fall (Dropped from Rest)", difficulty="medium",
                problem_text=f"A {subject} is dropped from rest. Find its speed after {t:g} s.",
                equation_id="eq_v_u_at", expression="v = u + a*t", law_node=LAW_ACCEL,
                knowns={"t": t}, target="v", trigger_phrases=["dropped from rest", "dropped"],
                constraints_fired=["meta_from_rest_u_zero", "meta_free_fall_a_minus_g"],
                implied_values={"u": 0.0, "a": -9.8},
                method=(
                    "Being dropped implies u = 0 and free fall implies a = -9.8 m/s^2, then use "
                    "the first equation of motion (v = u + at)."
                ),
                solution_steps=[
                    "Dropped from rest implies u = 0.",
                    "Free fall implies a = -9.8 m/s^2.",
                    "Substitute into v = u + a*t.",
                    "Compute the numeric result.",
                ],
            )

        _append_unique(cases, seen, build)

    for i in range(10):
        subject = subjects[(i + 1) % len(subjects)]

        def build(subject=subject, i=i):
            t = _rand_int(1, 8)
            return _case(
                f"suvat_free_fall_dropped_find_s_{i:03d}", grade=11, concept="Free Fall Under Gravity",
                subconcept="Free Fall (Dropped from Rest)", difficulty="medium",
                problem_text=(
                    f"A {subject} is dropped from rest and falls for {t:g} s. "
                    f"Find the distance it falls."
                ),
                equation_id="eq_s_ut_half_at2", expression="s = u*t + a*t**2/2", law_node=LAW_ACCEL,
                knowns={"t": t}, target="s", trigger_phrases=["dropped from rest", "dropped"],
                constraints_fired=["meta_from_rest_u_zero", "meta_free_fall_a_minus_g"],
                implied_values={"u": 0.0, "a": -9.8},
                method=(
                    "Being dropped implies u = 0 and free fall implies a = -9.8 m/s^2, then use "
                    "the second equation of motion (s = ut + 1/2 at^2)."
                ),
                solution_steps=[
                    "Dropped from rest implies u = 0.",
                    "Free fall implies a = -9.8 m/s^2.",
                    "Substitute into s = u*t + a*t^2/2.",
                    "Compute the numeric result.",
                ],
            )

        _append_unique(cases, seen, build)


def generate_time_to_max_height_cases(cases: list[dict], seen: set[str]) -> None:
    """At maximum height v=0 and a=-9.8 (both already-existing graph
    constraints); rearranging eq1 for t is linear and safe."""
    for i in range(10):

        def build(i=i):
            u = _rand(5, 30, 1)
            return _case(
                f"suvat_time_to_max_height_{i:03d}", grade=11, concept="Free Fall Under Gravity",
                subconcept="Time to Reach Maximum Height", difficulty="hard",
                problem_text=(
                    f"A ball is thrown upward with an initial velocity of {u:g} m/s. "
                    f"Find the time it takes to reach its maximum height."
                ),
                equation_id="eq_v_u_at", expression="v = u + a*t", law_node=LAW_ACCEL,
                knowns={"u": u}, target="t",
                trigger_phrases=["thrown upward", "maximum height"],
                constraints_fired=["constraint_max_height_v_zero", "constraint_free_fall_upward_a_minus_g"],
                implied_values={"v": 0.0, "a": -9.8},
                method=(
                    "Apply the zero-velocity-at-maximum-height constraint (v = 0) and the upward "
                    "free-fall constraint (a = -9.8 m/s^2), then rearrange the first equation of "
                    "motion (v = u + at) to solve for t."
                ),
                solution_steps=[
                    "Maximum height implies v = 0.",
                    "Upward free fall implies a = -9.8 m/s^2.",
                    "Rearrange v = u + a*t to t = -u/a.",
                    "Compute the numeric result.",
                ],
            )

        _append_unique(cases, seen, build)


def generate_velocity_at_height_cases(cases: list[dict], seen: set[str]) -> None:
    """Velocity partway up a vertical throw: only the upward free-fall
    constraint applies (not max-height), s kept below max height so
    v^2 = u^2 - 19.6*s stays non-negative."""
    for i in range(10):

        def build(i=i):
            u = _rand(10, 30, 1)
            max_height = (u**2) / 19.6
            s = round(max_height * random.uniform(0.3, 0.75), 2)
            return _case(
                f"suvat_velocity_at_height_{i:03d}", grade=11, concept="Free Fall Under Gravity",
                subconcept="Velocity at a Given Height", difficulty="hard",
                problem_text=(
                    f"A ball is thrown upward with an initial velocity of {u:g} m/s. "
                    f"Find its speed when it has risen {s:g} m."
                ),
                equation_id="eq_v2_u2_2as", expression="v**2 = u**2 + 2*a*s", law_node=LAW_ACCEL,
                knowns={"u": u, "s": s}, target="v",
                trigger_phrases=["thrown upward"],
                constraints_fired=["constraint_free_fall_upward_a_minus_g"],
                implied_values={"a": -9.8},
                method=(
                    "Apply the upward free-fall constraint (a = -9.8 m/s^2), then use the third "
                    "equation of motion (v^2 = u^2 + 2as), taking the positive root."
                ),
                solution_steps=[
                    "Upward free fall implies a = -9.8 m/s^2.",
                    "Substitute into v^2 = u^2 + 2*a*s.",
                    "Take the positive square root.",
                ],
            )

        _append_unique(cases, seen, build)


def generate_time_of_flight_cases(cases: list[dict], seen: set[str]) -> None:
    """New equation eq_time_of_flight (t = -2u/a): total time aloft for a
    vertical throw that returns to its launch height."""
    for i in range(10):

        def build(i=i):
            u = _rand(5, 30, 1)
            return _case(
                f"suvat_time_of_flight_{i:03d}", grade=11, concept="Free Fall Under Gravity",
                subconcept="Time of Flight", difficulty="hard",
                problem_text=(
                    f"A ball is thrown upward with an initial velocity of {u:g} m/s. "
                    f"Find the total time it stays in the air before returning to the same height."
                ),
                equation_id="eq_time_of_flight", expression="t = -2*u/a", law_node=LAW_ACCEL,
                knowns={"u": u}, target="t",
                trigger_phrases=["thrown upward", "time of flight", "returning to the same height"],
                constraints_fired=["constraint_free_fall_upward_a_minus_g"],
                implied_values={"a": -9.8},
                method=(
                    "Apply the upward free-fall constraint (a = -9.8 m/s^2), then use the "
                    "time-of-flight equation (t = -2u/a) for a projectile returning to its "
                    "launch height."
                ),
                solution_steps=[
                    "Upward free fall implies a = -9.8 m/s^2.",
                    "Substitute into t = -2*u/a.",
                    "Compute the numeric result.",
                ],
            )

        _append_unique(cases, seen, build)


def generate_avg_velocity_cases(cases: list[dict], seen: set[str]) -> None:
    """avg_v = (u+v)/2 - average velocity under uniform acceleration."""
    subjects = ["car", "cyclist", "train", "runner", "motorbike"]

    for i in range(10):
        subject = subjects[i % len(subjects)]

        def build(subject=subject, i=i):
            u = _rand(0, 20, 1)
            v = _rand(5, 40, 1)
            return _case(
                f"suvat_avgvel_find_avgv_{i:03d}", grade=11, concept="Uniform Acceleration (SUVAT)",
                subconcept="Average Velocity", difficulty="medium",
                problem_text=(
                    f"A {subject} accelerates uniformly from {u:g} m/s to {v:g} m/s. "
                    f"Find its average velocity over this interval."
                ),
                equation_id="eq_avg_velocity", expression="avg_v = (u+v)/2", law_node=LAW_ACCEL,
                knowns={"u": u, "v": v}, target="avg_v",
                trigger_phrases=["accelerates uniformly", "average velocity"],
                constraints_fired=[], implied_values={},
                method=(
                    "For uniform acceleration, average velocity is the mean of initial and final "
                    "velocity: avg_v = (u+v)/2."
                ),
                solution_steps=[
                    "Identify knowns u, v.",
                    "Substitute into avg_v = (u+v)/2.",
                    "Compute the numeric result.",
                ],
            )

        _append_unique(cases, seen, build)

    for i in range(8):
        subject = subjects[(i + 1) % len(subjects)]

        def build(subject=subject, i=i):
            avg_v = _rand(5, 30, 1)
            u = _rand(0, avg_v, 1)
            return _case(
                f"suvat_avgvel_find_v_{i:03d}", grade=11, concept="Uniform Acceleration (SUVAT)",
                subconcept="Average Velocity", difficulty="medium",
                problem_text=(
                    f"A {subject} starts at {u:g} m/s under uniform acceleration and has an "
                    f"average velocity of {avg_v:g} m/s over an interval. Find its final velocity."
                ),
                equation_id="eq_avg_velocity", expression="avg_v = (u+v)/2", law_node=LAW_ACCEL,
                knowns={"u": u, "avg_v": avg_v}, target="v", trigger_phrases=["average velocity"],
                constraints_fired=[], implied_values={},
                method="Rearrange avg_v = (u+v)/2 to solve for v.",
                solution_steps=[
                    "Identify knowns u, avg_v.",
                    "Rearrange avg_v = (u+v)/2 to v = 2*avg_v - u.",
                    "Compute the numeric result.",
                ],
            )

        _append_unique(cases, seen, build)


def generate_relative_speed_cases(cases: list[dict], seen: set[str]) -> None:
    """eq_relative_speed: relative_speed = v1 + direction*v2, where the
    direction constraint (+1 opposite / -1 same direction) disambiguates
    which physical scenario applies - two bodies never compete for the
    same target the way two separate equations with identical symbol sets
    would."""
    pairs = [("car", "truck"), ("cyclist", "runner"), ("train", "bus"), ("scooter", "van"), ("bike", "jogger")]

    for i in range(8):
        a_name, b_name = pairs[i % len(pairs)]

        def build(a_name=a_name, b_name=b_name, i=i):
            v1 = _rand(5, 25, 1)
            v2 = _rand(5, 25, 1)
            return _case(
                f"relmotion_opp_find_relspeed_{i:03d}", grade=11, concept="Relative Motion",
                subconcept="Relative Velocity (Opposite Directions)", difficulty="medium",
                problem_text=(
                    f"A {a_name} moves at {v1:g} m/s and a {b_name} moves at {v2:g} m/s, "
                    f"travelling towards each other. Find their relative velocity."
                ),
                equation_id="eq_relative_speed", expression="relative_speed = v1 + direction*v2",
                law_node=LAW_RELATIVE,
                knowns={"v1": v1, "v2": v2}, target="relative_speed",
                trigger_phrases=["towards each other"],
                constraints_fired=["constraint_opposite_direction_relative_motion"],
                implied_values={"direction": 1.0},
                method=(
                    "The opposite-direction constraint implies direction = +1, then "
                    "relative_speed = v1 + direction*v2."
                ),
                solution_steps=[
                    "Opposite directions imply direction = +1.",
                    "Substitute into relative_speed = v1 + direction*v2.",
                    "Compute the numeric result.",
                ],
            )

        _append_unique(cases, seen, build)

    for i in range(8):
        a_name, b_name = pairs[(i + 1) % len(pairs)]

        def build(a_name=a_name, b_name=b_name, i=i):
            v2 = _rand(5, 20, 1)
            v1 = _rand(v2 + 1, v2 + 20, 1)
            return _case(
                f"relmotion_same_find_relspeed_{i:03d}", grade=11, concept="Relative Motion",
                subconcept="Relative Velocity (Same Direction)", difficulty="medium",
                problem_text=(
                    f"A {a_name} moves at {v1:g} m/s and a {b_name} ahead of it moves at {v2:g} m/s, "
                    f"both in the same direction. Find the relative velocity of the {a_name} with "
                    f"respect to the {b_name}."
                ),
                equation_id="eq_relative_speed", expression="relative_speed = v1 + direction*v2",
                law_node=LAW_RELATIVE,
                knowns={"v1": v1, "v2": v2}, target="relative_speed",
                trigger_phrases=["same direction"],
                constraints_fired=["constraint_same_direction_relative_motion"],
                implied_values={"direction": -1.0},
                method=(
                    "The same-direction constraint implies direction = -1, then "
                    "relative_speed = v1 + direction*v2."
                ),
                solution_steps=[
                    "Same direction implies direction = -1.",
                    "Substitute into relative_speed = v1 + direction*v2.",
                    "Compute the numeric result.",
                ],
            )

        _append_unique(cases, seen, build)

    for i in range(6):
        a_name, b_name = pairs[(i + 2) % len(pairs)]

        def build(a_name=a_name, b_name=b_name, i=i):
            v2 = _rand(5, 20, 1)
            relative_speed = _rand(15, 45, 1)
            return _case(
                f"relmotion_opp_find_v1_{i:03d}", grade=11, concept="Relative Motion",
                subconcept="Relative Velocity (Opposite Directions)", difficulty="hard",
                problem_text=(
                    f"A {a_name} and a {b_name} travel towards each other. The {b_name} moves at "
                    f"{v2:g} m/s and their relative velocity is {relative_speed:g} m/s. "
                    f"Find the speed of the {a_name}."
                ),
                equation_id="eq_relative_speed", expression="relative_speed = v1 + direction*v2",
                law_node=LAW_RELATIVE,
                knowns={"v2": v2, "relative_speed": relative_speed}, target="v1",
                trigger_phrases=["towards each other"],
                constraints_fired=["constraint_opposite_direction_relative_motion"],
                implied_values={"direction": 1.0},
                method=(
                    "The opposite-direction constraint implies direction = +1; rearrange "
                    "relative_speed = v1 + direction*v2 to solve for v1."
                ),
                solution_steps=[
                    "Opposite directions imply direction = +1.",
                    "Rearrange relative_speed = v1 + direction*v2 to v1 = relative_speed - direction*v2.",
                    "Compute the numeric result.",
                ],
            )

        _append_unique(cases, seen, build)


def _relative_motion_chain_case(
    case_id: str,
    *,
    subconcept: str,
    difficulty: str,
    problem_text: str,
    knowns: dict[str, float],
    direction: float,
    target: str,
    constraint_id: str,
    trigger_phrases: list[str],
    method: str,
    solution_steps: list[str],
) -> dict:
    """Two-equation chain (eq_relative_speed -> eq_meeting_time): relative_speed
    is an intermediate derived quantity, not a raw known, so this bypasses the
    single-equation _case() helper and derives both steps directly."""
    values: dict[str, float] = {**knowns, "direction": direction}
    values["relative_speed"] = values["v1"] + direction * values["v2"]
    if target == "t":
        values["t"] = values["separation"] / values["relative_speed"]
    elif target == "separation":
        values["separation"] = values["t"] * values["relative_speed"]
    value = round(values[target], 4)
    return {
        "case_id": case_id,
        "chapter": "kinematics",
        "grade": 11,
        "concept": "Relative Motion",
        "subconcept": subconcept,
        "difficulty": difficulty,
        "problem_text": problem_text,
        "known_symbols": sorted(knowns),
        "derived_symbols": ["direction", "relative_speed"],
        "unknown": target,
        "trigger_phrases": trigger_phrases,
        "constraints_fired": [constraint_id],
        "equations_used": ["eq_relative_speed", "eq_meeting_time"],
        "law_nodes": [LAW_RELATIVE],
        "implied_values": {"direction": direction},
        "method": method,
        "solution_steps": solution_steps,
        "final_answer": {"value": value, "unit": UNIT_BY_SYMBOL[target]},
    }


def generate_meeting_time_cases(cases: list[dict], seen: set[str]) -> None:
    pairs = [("car", "truck"), ("cyclist", "runner"), ("train", "bus"), ("scooter", "van")]

    for i in range(10):
        a_name, b_name = pairs[i % len(pairs)]

        def build(a_name=a_name, b_name=b_name, i=i):
            v1 = _rand(5, 25, 1)
            v2 = _rand(5, 25, 1)
            separation = _rand_int(50, 900)
            return _relative_motion_chain_case(
                f"relmotion_meeting_time_{i:03d}", subconcept="Time to Meet", difficulty="hard",
                problem_text=(
                    f"A {a_name} and a {b_name} are {separation:g} m apart and start moving towards "
                    f"each other at {v1:g} m/s and {v2:g} m/s respectively. Find the time after "
                    f"which they meet."
                ),
                knowns={"v1": v1, "v2": v2, "separation": separation}, direction=1.0, target="t",
                constraint_id="constraint_opposite_direction_relative_motion",
                trigger_phrases=["towards each other"],
                method=(
                    "Opposite directions imply direction = +1; find the relative velocity "
                    "(relative_speed = v1 + direction*v2), then meeting time = separation / "
                    "relative_speed."
                ),
                solution_steps=[
                    "Opposite directions imply direction = +1.",
                    "Compute relative_speed = v1 + direction*v2.",
                    "Compute t = separation / relative_speed.",
                ],
            )

        _append_unique(cases, seen, build)


def generate_catch_up_time_cases(cases: list[dict], seen: set[str]) -> None:
    pairs = [("car", "truck"), ("cyclist", "runner"), ("train", "bus"), ("scooter", "van")]

    for i in range(10):
        a_name, b_name = pairs[i % len(pairs)]

        def build(a_name=a_name, b_name=b_name, i=i):
            v2 = _rand(5, 20, 1)
            v1 = _rand(v2 + 1, v2 + 15, 1)
            separation = _rand_int(50, 900)
            return _relative_motion_chain_case(
                f"relmotion_catchup_time_{i:03d}", subconcept="Time to Catch Up", difficulty="hard",
                problem_text=(
                    f"A {b_name} is {separation:g} m ahead of a {a_name}, both moving in the same "
                    f"direction. The {a_name} travels at {v1:g} m/s and the {b_name} at {v2:g} m/s. "
                    f"Find the time it takes the {a_name} to catch up to the {b_name}."
                ),
                knowns={"v1": v1, "v2": v2, "separation": separation}, direction=-1.0, target="t",
                constraint_id="constraint_same_direction_relative_motion",
                trigger_phrases=["same direction", "catch up"],
                method=(
                    "Same direction implies direction = -1; find the relative velocity "
                    "(relative_speed = v1 + direction*v2), then catch-up time = separation / "
                    "relative_speed."
                ),
                solution_steps=[
                    "Same direction implies direction = -1.",
                    "Compute relative_speed = v1 + direction*v2.",
                    "Compute t = separation / relative_speed.",
                ],
            )

        _append_unique(cases, seen, build)


def main() -> None:
    cases: list[dict] = []
    seen_texts: set[str] = set()
    generate_eq_v_u_at(cases, seen_texts)
    generate_eq_s_vt(cases, seen_texts)
    generate_eq_average_speed(cases, seen_texts)
    generate_eq_s_ut_half_at2(cases, seen_texts)
    generate_eq_v2_u2_2as(cases, seen_texts)
    generate_from_rest_cases(cases, seen_texts)
    generate_comes_to_rest_cases(cases, seen_texts)
    generate_max_height_cases(cases, seen_texts)
    generate_eq2_additional_targets(cases, seen_texts)
    generate_free_fall_dropped_cases(cases, seen_texts)
    generate_time_to_max_height_cases(cases, seen_texts)
    generate_velocity_at_height_cases(cases, seen_texts)
    generate_time_of_flight_cases(cases, seen_texts)
    generate_avg_velocity_cases(cases, seen_texts)
    generate_relative_speed_cases(cases, seen_texts)
    generate_meeting_time_cases(cases, seen_texts)
    generate_catch_up_time_cases(cases, seen_texts)

    case_ids = [case["case_id"] for case in cases]
    assert len(case_ids) == len(set(case_ids)), "duplicate case_id generated"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case) + "\n")

    print(f"Wrote {len(cases)} cases to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
