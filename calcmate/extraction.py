from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from typing import Any

from calcmate.constants import (
    CANONICAL_SYMBOLS,
    SYMBOL_ALIASES,
    SYMBOL_ALIAS_PATTERNS,
    UNIT_BY_SYMBOL,
)
from calcmate.models import ExtractedProblem, Quantity
from calcmate.text_patterns import compile_phrase, compile_phrase_set, phrase_to_pattern


SUPPORTED_SYMBOLS = set(UNIT_BY_SYMBOL)

_GRADE_6_7_FAMILY = {"speed", "distance", "time"}
_SUVAT_FAMILY = {"u", "v", "a", "t", "s"}
# eq_average_speed (the only grade 6/7 equation) has no acceleration term -
# it's inherently a uniform-motion/no-acceleration model - so a stray "u"
# (initial velocity) or "v" (final velocity) both just mean "the speed" in
# that context. "v" is checked first since it's the more common LLM label
# for "the" velocity in a single-speed problem; "u" only maps if "speed"
# is still unclaimed.
_SUVAT_TO_GRADE_6_7 = {"v": "speed", "u": "speed", "s": "distance", "t": "time"}
# Reverse direction: a stray grade 6/7 "speed" reconciled into a SUVAT-family
# problem *as a known quantity* almost always describes the given speed at
# the start of the problem ("moving at 25 m/s", "starts from rest") rather
# than the final velocity being solved for - the final velocity is either
# the target itself (so it shouldn't also appear as a known) or supplied
# separately by a constraint (comes-to-rest, max-height). Mapping it to "u"
# instead of "v" is what makes braking/free-fall problems phrased with a
# plain "speed" solvable instead of silently asserting the wrong final
# velocity as already known. This is deliberately a different mapping from
# _GRADE_6_7_TARGET_TO_SUVAT below - the same word "speed" means the given
# "u" when it's a known, but the sought "v" when it's the target.
_GRADE_6_7_KNOWN_TO_SUVAT = {"speed": "u", "distance": "s", "time": "t"}

# Used only to convert the *target* symbol: "find its speed" under
# acceleration means "find the final velocity", i.e. v - not u.
_GRADE_6_7_TARGET_TO_SUVAT = {"speed": "v", "distance": "s", "time": "t"}

# Evidence that a problem involves acceleration (SUVAT), even when it's
# phrased with plain Grade 6/7 vocabulary ("find its speed", "find the
# distance", "how long does it take"). TARGET_PATTERNS below would otherwise
# route these onto the constant-speed-only Grade 6/7 equation, which has no
# way to represent a changing speed at all.
_ACCELERATION_EVIDENCE_PATTERN = compile_phrase_set(
    ["accelerat", "decelerat", "brak", "free fall", "thrown upward", "thrown up", "dropped"]
)
_ACCELERATION_UNIT_RE = re.compile(
    r"m/s\^?2|m/s²|met(?:er|re)s?\s*per\s*second\s*squared", re.IGNORECASE
)


def _has_acceleration_evidence(text: str) -> bool:
    lowered = text.lower()
    return bool(_ACCELERATION_EVIDENCE_PATTERN.search(lowered) or _ACCELERATION_UNIT_RE.search(lowered))

# Base accepted trigger phrases for Grade 6/7 uniform-motion wording. Each is
# compiled into a regex that also tolerates plural/verb-tense variants
# ("travels"/"travelled"/"travelling") instead of requiring an exact
# substring match against one accepted spelling.
GRADE_6_7_TRIGGER_PHRASES = [
    "average speed",
    "uniform motion",
    "constant speed",
    "starts from rest",
    "comes to rest",
    "travels",
    "covers",
    "journey",
    "walks",
    "runs",
    "moves",
]
_GRADE_6_7_TRIGGER_PATTERNS = [
    (phrase, compile_phrase(phrase)) for phrase in GRADE_6_7_TRIGGER_PHRASES
]

# Grade 6/7 questions ask for a plain quantity ("speed", "distance", "time")
# rather than a SUVAT symbol. Instead of enumerating every verb + phrasing
# combination ("find speed", "calculate the speed", "what is its speed", ...)
# as its own literal string, compose a small set of question-verb patterns
# so new phrasings ("determine the speed", "compute distance") are matched
# automatically.
_QUESTION_VERB = r"(?:find|calculate|compute|determine|what(?:'s|\s+is))"
TARGET_PATTERNS: list[tuple["re.Pattern[str]", str]] = [
    (re.compile(rf"\b{_QUESTION_VERB}\s+(?:its|the)?\s*speed\b", re.IGNORECASE), "speed"),
    (re.compile(rf"\b{_QUESTION_VERB}\s+(?:its|the)?\s*distance\b", re.IGNORECASE), "distance"),
    (re.compile(r"\bhow\s+far\b", re.IGNORECASE), "distance"),
    (re.compile(rf"\b{_QUESTION_VERB}\s+(?:its|the)?\s*time\b", re.IGNORECASE), "time"),
    (re.compile(r"\bhow\s+long\b", re.IGNORECASE), "time"),
]

# Unit strings coming back from the LLM vary in spelling/spacing
# ("meters/second", "m per s", "seconds", "sec"). Normalize with regex
# substitutions instead of chaining exact-string .replace() calls for every
# spelling, so unlisted variants still collapse to the canonical unit.
_UNIT_NORMALIZATION_RULES: list[tuple["re.Pattern[str]", str]] = [
    (re.compile(r"\bper\b"), "/"),
    (re.compile(r"met(?:er|re)s?\b"), "m"),
    (re.compile(r"kilomet(?:er|re)s?\b"), "km"),
    (re.compile(r"sec(?:ond)?s?\b"), "s"),
    (re.compile(r"\bhrs?\b"), "h"),
    (re.compile(r"\bhours?\b"), "h"),
    (re.compile(r"\bmin(?:ute)?s?\b"), "min"),
    (re.compile(r"\bkmph\b"), "km/h"),
    (re.compile(r"squared\b"), "^2"),
    (re.compile(r"\s+"), ""),
    (re.compile(r"(?<=/m)2\b"), "^2"),
    (re.compile(r"(?<=/s)2\b"), "^2"),
]

# Finds the unit text trailing the last number in a source_text quote, e.g.
# "initial velocity 20 m/s" -> "m/s", "acceleration -10 m/s^2" -> "m/s^2",
# "20 minutes" -> "minutes". Anchored at the end of the string since a
# quoted value/unit pair is expected to end the quoted phrase.
_NUMBER_THEN_UNIT_RE = re.compile(r"-?\d+\.?\d*\s*([a-zA-Z][a-zA-Z/\^0-9]*)\s*$")

# Same anchor as above, but captures the number too - used to recover a
# quantity that the LLM quoted in source_text but forgot to add to
# knowns_raw. Unit group is optional since a bare quoted number is still
# recoverable (falls back to the symbol's canonical SI unit).
_NUMBER_AND_UNIT_RE = re.compile(r"(-?\d+\.?\d*)\s*([a-zA-Z][a-zA-Z/\^0-9]*)?\s*$")

# The trailing duration clause ("... for 5 s", "... in 6 s", "... after 4 s")
# is the single quantity the LLM most often drops from knowns_raw even
# though every other known was extracted correctly. Recovered directly from
# the raw problem text as a last resort, independent of whatever the LLM
# returned, rather than reporting under-constrained over one omitted value.
_DURATION_PHRASE_RE = re.compile(
    r"\b(?:for|in|after|over)\s+(-?\d+\.?\d*)\s*(seconds?|secs?|s|minutes?|mins?|min|hours?|hrs?|h)\b",
    re.IGNORECASE,
)

# Compact "symbol = value unit" shorthand (e.g. "t=5sec", "u = 20 m/s") shows
# up in problems typed as a list of givens rather than a sentence. The small
# Groq model reliably reads sentence-style phrasing ("for 5 s") but sometimes
# drops this notation from knowns_raw and source_text entirely - unlike the
# duration/quoted-value recoveries above, there's then nothing left to
# recover *from*, so this is matched directly against the raw problem text
# instead. One pattern per canonical symbol (built from the same accepted
# aliases used elsewhere, e.g. "initial velocity" for "u"), requiring the
# right-hand side to start with a digit so it can't misfire on an actual
# equation quoted in the problem text (e.g. "v = u + a*t").
def _build_shorthand_assignment_patterns() -> dict[str, "re.Pattern[str]"]:
    patterns: dict[str, "re.Pattern[str]"] = {}
    for symbol in CANONICAL_SYMBOLS:
        aliases = SYMBOL_ALIASES.get(symbol, set()) | {symbol}
        alternation = "|".join(sorted(filter(None, (phrase_to_pattern(alias) for alias in aliases))))
        if not alternation:
            continue
        patterns[symbol] = re.compile(
            rf"(?:{alternation})\s*=\s*(-?\d+\.?\d*)\s*([a-zA-Z][a-zA-Z/\^0-9]*)?",
            re.IGNORECASE,
        )
    return patterns


_SHORTHAND_ASSIGNMENT_PATTERNS = _build_shorthand_assignment_patterns()


class ExtractionError(ValueError):
    pass


class ExtractPhysicsProblemSignature:
    pass


def _build_signature():#This function creates the DSPy “contract” for the LLM.
    import dspy

    class ExtractPhysicsProblem(dspy.Signature):
        #A dspy.Signature tells DSPy:
        #These are my input fields.
        #These are my output fields.
        #These are the instructions for the LLM.
        """Extract only explicit facts and trigger phrases from a 1D kinematics problem.

        Do not solve the problem. Do not choose equations. Do not infer hidden physics
        Return strict JSON with this shape:
        {
          "knowns_raw": {"u": 20, "a": -10},
          "units": {"u": "m/s", "a": "m/s^2"},
          "source_text": {"u": "initial velocity 20 m/s", "a": "acceleration -10 m/s^2"},
          "unknown_hint": "s",
          "domain_hint": "kinematics",
          "matched_trigger_phrases": ["maximum height", "thrown upward"]
        }

        Symbols: u initial velocity, v final velocity, a acceleration,
        t time, s displacement/distance/height, distance, speed, time,
        avg_v average velocity, v1 velocity of the first object, v2 velocity
        of the second object, separation initial gap between two objects.
        For Grade 6 and Grade 7 uniform-motion questions, use plain targets:
        speed for "find speed", distance for "find distance", and time for
        "find time". Keep acceleration/SUVAT questions on u, v, a, t, s. For
        problems about two moving objects (relative velocity, time to meet,
        time to catch up), use v1 for the first object's speed, v2 for the
        second object's speed, and separation for the initial gap between
        them - do not compute the relative velocity yourself.
        Recognize trigger phrases such as average speed, uniform motion,
        constant speed, starts from rest, comes to rest, travels, covers,
        journey, walks, runs, moves, towards each other, opposite directions,
        same direction, chasing, and catching up.
        Trigger phrases must be copied from phrases present in the problem text.
        Do not infer hidden physics values such as v=0 at maximum height.
        Do not convert units. Report each value exactly as it appears in the
        source text, with its original unit unchanged - e.g. "5 km" must stay
        knowns_raw value 5 with unit "km" (not converted to 5000/"m"), and
        "20 minutes" must stay value 20 with unit "minutes" (not "s"). Unit
        conversion is handled separately downstream; converting here, even
        partially or for only some quantities, produces wrong answers.
        """

        problem_text: str = dspy.InputField()# how the input will be 
        extraction_json: str = dspy.OutputField(desc="Strict JSON only, no markdown.")#how the output will be 

    return ExtractPhysicsProblem #This returns the signature class so we can plug it into DSPy.


class DSPyPhysicsExtractor:
    """Layer 1. DSPy + LLM extraction boundary; no physics reasoning happens here."""
    #dspy and llm link
    def __init__(self) -> None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ExtractionError(
                "Missing GROQ_API_KEY. Create a Groq API key, then set it before starting the server."
            )

        import dspy

        model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
        api_base = os.environ.get("GROQ_API_BASE", "https://api.groq.com/openai/v1")
        lm = dspy.LM(
            f"openai/{model}",
            api_key=api_key,
            api_base=api_base,
            temperature=0.1,
            max_tokens=700,
        )
        dspy.configure(lm=lm)
        self.extract_program = dspy.Predict(_build_signature())#So now self.extract_program behaves like a Python function:

    #this fn is called wen executing solve in the pipeline,after the overlay is loaded
    def extract(self, text: str) -> ExtractedProblem:
        prediction = self.extract_program(problem_text=text)#feeding input to dspy
        raw_json = getattr(prediction, "extraction_json", "")#Get prediction.extraction_json if it exists.If it does not exist, use "" instead.
        data = self._parse_json(raw_json)#cleaning the output and converting to a dict style
        return self._to_problem(text, data)#format the answer into ExtractedProblem class 

    def _parse_json(self, raw_json: str) -> dict[str, Any]:
        cleaned = raw_json.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.removeprefix("json").strip()
        try:
            data = json.loads(cleaned)#json format is returned but like a string so you convert that into a python object
        except json.JSONDecodeError as exc:
            raise ExtractionError(f"LLM extraction was not valid JSON: {raw_json}") from exc
        if not isinstance(data, dict):
            raise ExtractionError("LLM extraction must be a JSON object.")
        return data

    def _to_problem(self, text: str, data: dict[str, Any]) -> ExtractedProblem:
        target = self._normalize_target(text, str(data.get("unknown_hint") or data.get("target") or ""))
        if target not in SUPPORTED_SYMBOLS:
            raise ExtractionError(f"Unsupported or missing target symbol: {target!r}")

        quantities: dict[str, Quantity] = {}
        if "knowns_raw" in data:
            units = data.get("units", {})
            source_text = data.get("source_text", {})
            for symbol, value in data.get("knowns_raw", {}).items():
                normalized_symbol = str(symbol).strip().lower()
                if normalized_symbol not in SUPPORTED_SYMBOLS:
                    continue
                raw_source_text = str(source_text.get(symbol, ""))
                claimed_unit = str(units.get(symbol, UNIT_BY_SYMBOL.get(normalized_symbol, "")))
                quantities[normalized_symbol] = Quantity(
                    symbol=normalized_symbol,
                    value=float(value),
                    unit=self._resolve_unit(claimed_unit, raw_source_text),
                    source_text=raw_source_text,
                )
            # The LLM sometimes quotes a value in source_text but forgets to
            # also add it to knowns_raw (e.g. it correctly writes
            # source_text["t"] = "6 s" but never emits knowns_raw["t"]).
            # Recover anything quoted-but-missing directly from the quote
            # instead of silently losing it and reporting under-constrained.
            for symbol, quote in source_text.items():
                normalized_symbol = self._normalize_symbol(str(symbol))
                if normalized_symbol not in SUPPORTED_SYMBOLS or normalized_symbol in quantities:
                    continue
                recovered = self._recover_quantity_from_source_text(normalized_symbol, str(quote))
                if recovered is not None:
                    quantities[normalized_symbol] = recovered
        for item in data.get("quantities", []):
            symbol = str(item.get("symbol", "")).strip().lower()
            if symbol not in SUPPORTED_SYMBOLS:
                continue
            try:
                value = float(item["value"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ExtractionError(f"Invalid quantity for {symbol!r}: {item}") from exc
            raw_source_text = str(item.get("source_text", ""))
            quantities[symbol] = Quantity(
                symbol=symbol,
                value=value,
                unit=self._resolve_unit(str(item.get("unit", "")), raw_source_text),
                source_text=raw_source_text,
            )

        quantities = self._recover_shorthand_assignments(text, quantities)
        quantities = self._recover_missing_duration(text, target, quantities)
        quantities = self._reconcile_symbol_family(quantities, target)

        raw_triggers = data.get("matched_trigger_phrases", data.get("trigger_phrases", []))
        triggers = [str(phrase).strip().lower() for phrase in raw_triggers if str(phrase).strip()]
        domain_hint = str(data.get("domain_hint", "kinematics")).strip().lower() or "kinematics"
        return ExtractedProblem(
            raw_text=text,
            quantities=quantities,
            target=target,
            trigger_phrases=triggers,
            domain_hint=domain_hint,
        )

    def _recover_shorthand_assignments(
        self, text: str, quantities: dict[str, Quantity]
    ) -> dict[str, Quantity]:
        recovered = dict(quantities)
        for symbol, pattern in _SHORTHAND_ASSIGNMENT_PATTERNS.items():
            if symbol in recovered:
                continue
            match = pattern.search(text)
            if not match:
                continue
            value = float(match.group(1))
            unit_text = match.group(2)
            unit = self._normalize_unit(unit_text) if unit_text else UNIT_BY_SYMBOL.get(symbol, "")
            recovered[symbol] = Quantity(
                symbol=symbol, value=value, unit=unit, source_text=match.group(0).strip()
            )
        return recovered

    def _recover_missing_duration(
        self, text: str, target: str, quantities: dict[str, Quantity]
    ) -> dict[str, Quantity]:
        if target in _SUVAT_FAMILY:
            time_symbol = "t"
        elif target in _GRADE_6_7_FAMILY:
            time_symbol = "time"
        else:
            return quantities
        if time_symbol in quantities:
            return quantities

        match = _DURATION_PHRASE_RE.search(text)
        if not match:
            return quantities

        value = float(match.group(1))
        unit = self._normalize_unit(match.group(2))
        recovered = dict(quantities)
        recovered[time_symbol] = Quantity(
            symbol=time_symbol, value=value, unit=unit, source_text=match.group(0)
        )
        return recovered

    def _reconcile_symbol_family(
        self, quantities: dict[str, Quantity], target: str
    ) -> dict[str, Quantity]:
        # "t" (SUVAT time) and "time" (plain grade 6/7 time) are distinct
        # canonical symbols so grade 6/7 targets don't get folded into SUVAT
        # ones (see _normalize_target). But the LLM extracts each known
        # quantity independently, so it can land on "t" for one known and
        # "distance"/"speed" for another within the same problem - the mix
        # then satisfies no equation at all. Once the target says which
        # family this problem is in, remap any stray cross-family knowns
        # into that family instead of leaving them unusable.
        if target in _GRADE_6_7_FAMILY:
            remap = _SUVAT_TO_GRADE_6_7
        elif target in _SUVAT_FAMILY:
            remap = _GRADE_6_7_KNOWN_TO_SUVAT
        else:
            return quantities

        reconciled = dict(quantities)
        for old_symbol, new_symbol in remap.items():
            if old_symbol in reconciled and new_symbol not in reconciled:
                reconciled[new_symbol] = replace(reconciled.pop(old_symbol), symbol=new_symbol)
        return reconciled

    def _normalize_target(self, text: str, raw_target: str) -> str:
        # Grade 6/7 phrasing ("find speed", "how far", ...) asks for a plain
        # quantity, not a SUVAT symbol - matched directly, bypassing symbol
        # alias normalization so "speed" isn't folded into "v".
        lowered_text = text.lower()
        for pattern, target in TARGET_PATTERNS:
            if pattern.search(lowered_text):
                return self._suvat_if_accelerating(text, target)
        return self._suvat_if_accelerating(text, self._normalize_symbol(raw_target))

    def _suvat_if_accelerating(self, text: str, symbol: str) -> str:
        # A Grade 6/7 target ("speed"/"distance"/"time") is only valid for
        # constant-speed problems. If the text shows acceleration is in play,
        # the target actually needs the SUVAT equations (which have an "a"
        # term the Grade 6/7 equation lacks), so route it onto the
        # equivalent SUVAT symbol instead.
        if symbol in _GRADE_6_7_TARGET_TO_SUVAT and _has_acceleration_evidence(text):
            return _GRADE_6_7_TARGET_TO_SUVAT[symbol]
        return symbol

    def _normalize_symbol(self, symbol: str) -> str:
        normalized = symbol.strip().lower().replace("_", " ")
        if normalized in CANONICAL_SYMBOLS:
            return normalized

        for canonical, pattern in SYMBOL_ALIAS_PATTERNS.items():
            if pattern.fullmatch(normalized):
                return canonical

        return normalized

    def _augment_grade_6_7_triggers(self, text: str, triggers: list[str]) -> list[str]:
        seen = set(triggers)
        lowered_text = text.lower()
        for phrase, pattern in _GRADE_6_7_TRIGGER_PATTERNS:
            if phrase in seen:
                continue
            if pattern.search(lowered_text):
                triggers.append(phrase)
                seen.add(phrase)
        return triggers

    def _normalize_unit(self, unit: str) -> str:
        normalized = unit.strip().lower()
        for pattern, replacement in _UNIT_NORMALIZATION_RULES:
            normalized = pattern.sub(replacement, normalized)
        return normalized

    def _resolve_unit(self, claimed_unit: str, source_text: str) -> str:
        # The LLM's "units" field is occasionally inconsistent with what it
        # actually wrote in "source_text" - e.g. pre-converting "5 km" to
        # 5000/"m" while leaving "20 minutes" as 20/"s" instead of "min".
        # source_text is meant to literally quote the problem text, so a
        # unit parsed from it is more trustworthy than the separate claim;
        # prefer it whenever the two disagree.
        from_quote = self._unit_from_source_text(source_text)
        normalized_claim = self._normalize_unit(claimed_unit)
        if from_quote and from_quote != normalized_claim:
            return from_quote
        return normalized_claim

    def _unit_from_source_text(self, source_text: str) -> str | None:
        match = _NUMBER_THEN_UNIT_RE.search(source_text)
        if not match:
            return None
        unit_text = match.group(1).strip()
        if not unit_text:
            return None
        return self._normalize_unit(unit_text)

    def _recover_quantity_from_source_text(self, symbol: str, quote: str) -> Quantity | None:
        match = _NUMBER_AND_UNIT_RE.search(quote)
        if not match:
            return None
        try:
            value = float(match.group(1))
        except ValueError:
            return None
        unit_text = match.group(2)
        unit = self._normalize_unit(unit_text) if unit_text else UNIT_BY_SYMBOL.get(symbol, "")
        return Quantity(symbol=symbol, value=value, unit=unit, source_text=quote)
