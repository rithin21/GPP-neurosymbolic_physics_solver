from __future__ import annotations

"""Reasoning-program VM.

Executes a ``reasoning_program`` (the opcode list defined in the dataset schema)
op-by-op against the knowledge graph, using Pint for unit conversion and SymPy
for equation solving. It is the deterministic engine that (a) replays dataset
cases to prove each program reproduces its ``final_answer`` and (b) will later
back the live planner->execute path.

Opcodes: identify_law, convert_units, apply_constraint, solve_equation, verify.

The VM reads all physics knowledge (SI units, dimensions, equation expressions,
constraint implications) from the graph -- no hard-coded per-symbol tables.
"""

from dataclasses import dataclass, field
from typing import Any

import sympy as sp

from calcmate.knowledge_graph import PhysicsKnowledgeGraph, load_default_graph
from calcmate.units import convert

# SymPy symbols the kinematics equations may reference.
_SYMPY_LOCALS = {name: sp.Symbol(name) for name in ("u", "v", "a", "t", "s", "distance", "speed", "time")}


class ExecError(Exception):
    """A reasoning program could not be executed (invalid op, missing known, ...)."""


@dataclass
class ExecResult:
    ok: bool
    computed_value: float | None
    computed_unit: str | None
    error: str | None = None
    trace: list[str] = field(default_factory=list)


class ProgramExecutor:
    def __init__(self, graph: PhysicsKnowledgeGraph | None = None) -> None:
        self.graph = graph or load_default_graph()
        self.si_unit: dict[str, str] = {}
        for _, attrs in self.graph.nodes_by_type("quantity"):
            symbol = attrs.get("symbol")
            if symbol:
                self.si_unit[symbol] = attrs.get("unit", "")
        self._laws = {node_id for node_id, _ in self.graph.nodes_by_type("law")}
        self._equations = {node_id: attrs for node_id, attrs in self.graph.nodes_by_type("equation")}
        self._constraints = {node_id: attrs for node_id, attrs in self.graph.nodes_by_type("constraint")}

    def execute(self, case: dict[str, Any], tolerance: float = 0.01) -> ExecResult:
        """Run a case's program; compare the resulting target value to final_answer."""
        state: dict[str, dict[str, Any]] = {}
        trace: list[str] = []

        for symbol, quantity in case.get("givens", {}).items():
            state[symbol] = {"value": float(quantity["value"]), "unit": quantity.get("unit", "")}

        target = case.get("target", {})
        target_symbol = target.get("symbol")

        try:
            for op in case.get("reasoning_program", []):
                self._step(op, state, trace)
        except ExecError as exc:
            return ExecResult(False, None, None, str(exc), trace)

        if target_symbol not in state:
            return ExecResult(False, None, None, f"target {target_symbol!r} was never solved", trace)

        computed = state[target_symbol]
        expected = case.get("final_answer", {})
        try:
            expected_value = float(expected["value"])
        except (KeyError, TypeError, ValueError):
            return ExecResult(False, round(computed["value"], 4), computed["unit"], "case has no numeric final_answer.value", trace)
        expected_unit = expected.get("unit", "")

        if not self._unit_match(computed["unit"], expected_unit):
            return ExecResult(
                False, round(computed["value"], 4), computed["unit"],
                f"unit mismatch: computed {computed['unit']!r} vs final_answer {expected_unit!r}", trace,
            )

        if abs(computed["value"] - expected_value) > max(tolerance, 0.005 * abs(expected_value)):
            return ExecResult(
                False, round(computed["value"], 4), computed["unit"],
                f"value mismatch: computed {round(computed['value'], 4)} vs final_answer {expected_value}", trace,
            )

        return ExecResult(True, round(computed["value"], 4), computed["unit"], None, trace)

    def _step(self, op: dict[str, Any], state: dict[str, dict[str, Any]], trace: list[str]) -> None:
        name = op.get("op")
        if name == "identify_law":
            law = op.get("law")
            if law not in self._laws:
                raise ExecError(f"identify_law references unknown law {law!r}")
            trace.append(f"identify_law {law}")

        elif name == "apply_constraint":
            constraint_id = op.get("constraint")
            if constraint_id not in self._constraints:
                raise ExecError(f"apply_constraint references unknown constraint {constraint_id!r}")
            implies = self._constraints[constraint_id].get("implies", {})
            for symbol, value in implies.items():
                state[symbol] = {"value": float(value), "unit": self.si_unit.get(symbol, "")}
            trace.append(f"apply_constraint {constraint_id} -> {implies}")

        elif name == "convert_units":
            symbol = op.get("symbol")
            to = op.get("to")
            if symbol not in state:
                raise ExecError(f"convert_units on symbol {symbol!r} which is not known yet")
            to_unit = self.si_unit.get(symbol, "") if to == "SI" else to
            current = state[symbol]
            new_value = convert(current["value"], current["unit"], to_unit)
            if new_value is None:
                raise ExecError(f"convert_units cannot convert {symbol} from {current['unit']!r} to {to_unit!r}")
            state[symbol] = {"value": new_value, "unit": to_unit}
            trace.append(f"convert_units {symbol} {current['unit']}->{to_unit} = {new_value:g}")

        elif name == "solve_equation":
            equation_id = op.get("equation")
            for_symbol = op.get("for")
            if equation_id not in self._equations:
                raise ExecError(f"solve_equation references unknown equation {equation_id!r}")
            value = self._solve(equation_id, for_symbol, state)
            state[for_symbol] = {"value": value, "unit": self.si_unit.get(for_symbol, "")}
            trace.append(f"solve_equation {equation_id} for {for_symbol} = {value:g}")

        elif name == "verify":
            trace.append("verify")

        else:
            raise ExecError(f"unknown op {name!r}")

    def _solve(self, equation_id: str, for_symbol: str, state: dict[str, dict[str, Any]]) -> float:
        attrs = self._equations[equation_id]
        symbols = list(attrs["symbols"])
        if for_symbol not in symbols:
            raise ExecError(f"{equation_id} does not contain symbol {for_symbol!r}")
        knowns = [symbol for symbol in symbols if symbol != for_symbol]
        missing = [symbol for symbol in knowns if symbol not in state]
        if missing:
            raise ExecError(f"{equation_id}: cannot solve for {for_symbol!r}, missing knowns {missing}")

        left_text, right_text = attrs["expression"].split("=", 1)
        equation = sp.Eq(
            sp.sympify(left_text.strip(), locals=_SYMPY_LOCALS),
            sp.sympify(right_text.strip(), locals=_SYMPY_LOCALS),
        )
        substitutions = {_SYMPY_LOCALS[symbol]: state[symbol]["value"] for symbol in knowns}
        roots = sp.solve(equation.subs(substitutions), _SYMPY_LOCALS[for_symbol])
        reals = [float(root) for root in roots if root.is_real]
        if not reals:
            raise ExecError(f"{equation_id}: no real solution for {for_symbol!r}")
        non_negative = [value for value in reals if value >= 0]
        return min(non_negative) if non_negative else reals[0]

    def _unit_match(self, actual: str, expected: str) -> bool:
        if (actual or "").strip() == (expected or "").strip():
            return True
        # Accept equivalent spellings (m/s == meter/second) but NOT different units
        # (m/s != km/h): a genuine unit is a factor-1.0 conversion of the other.
        factor = convert(1.0, actual, expected)
        return factor is not None and abs(factor - 1.0) < 1e-9
