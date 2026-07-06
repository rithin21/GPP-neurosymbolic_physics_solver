from __future__ import annotations

import json
import os
from typing import Any

from calcmate.constants import (
    CANONICAL_SYMBOLS,
    UNIT_BY_SYMBOL,
    SYMBOL_ALIASES,
)
from calcmate.models import ExtractedProblem, Quantity


SUPPORTED_SYMBOLS = set(UNIT_BY_SYMBOL)
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
TARGET_PHRASE_MAP = {
    "find speed": "speed",
    "find the speed": "speed",
    "calculate speed": "speed",
    "calculate the speed": "speed",
    "what is its speed": "speed",
    "what is the speed": "speed",
    "find distance": "distance",
    "find the distance": "distance",
    "calculate distance": "distance",
    "calculate the distance": "distance",
    "what distance": "distance",
    "how far": "distance",
    "find time": "time",
    "find the time": "time",
    "calculate time": "time",
    "calculate the time": "time",
    "how long": "time",
}


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
        t time, s displacement/distance/height, distance, speed, time.
        For Grade 6 and Grade 7 uniform-motion questions, use plain targets:
        speed for "find speed", distance for "find distance", and time for
        "find time". Keep acceleration/SUVAT questions on u, v, a, t, s.
        Recognize trigger phrases such as average speed, uniform motion,
        constant speed, starts from rest, comes to rest, travels, covers,
        journey, walks, runs, and moves.
        Trigger phrases must be copied from phrases present in the problem text.
        Do not infer hidden physics values such as v=0 at maximum height.
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
        if target not in SUPPORTED_SYMBOLS:#in physics i can use any symbol right then how is it going to handle thoses cases and isnt this really breakable?
            raise ExtractionError(f"Unsupported or missing target symbol: {target!r}")

        quantities: dict[str, Quantity] = {}
        if "knowns_raw" in data:
            units = data.get("units", {})
            source_text = data.get("source_text", {})
            for symbol, value in data.get("knowns_raw", {}).items():
                normalized_symbol = self._normalize_symbol(str(symbol))
                if normalized_symbol not in SUPPORTED_SYMBOLS:
                    continue
                quantities[normalized_symbol] = Quantity(
                    symbol=normalized_symbol,
                    value=float(value),
                    unit=self._normalize_unit(str(units.get(symbol, UNIT_BY_SYMBOL.get(normalized_symbol, "")))),
                    source_text=str(source_text.get(symbol, "")),
                )
        for item in data.get("quantities", []):
            symbol = self._normalize_symbol(str(item.get("symbol", "")))
            if symbol not in SUPPORTED_SYMBOLS:
                continue
            try:
                value = float(item["value"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ExtractionError(f"Invalid quantity for {symbol!r}: {item}") from exc
            quantities[symbol] = Quantity(
                symbol=symbol,
                value=value,
                unit=self._normalize_unit(str(item.get("unit", ""))),
                source_text=str(item.get("source_text", "")),
            )

        raw_triggers = data.get("matched_trigger_phrases", data.get("trigger_phrases", []))
        triggers = [str(phrase).strip().lower() for phrase in raw_triggers if str(phrase).strip()]
        triggers = self._augment_grade_6_7_triggers(text, triggers)
        domain_hint = str(data.get("domain_hint", "kinematics")).strip().lower() or "kinematics"
        return ExtractedProblem(
            raw_text=text,
            quantities=quantities,
            target=target,
            trigger_phrases=triggers,
            domain_hint=domain_hint,
        )

    def _normalize_target(self, text: str, raw_target: str) -> str:
        lowered_text = text.lower()
        for phrase, target in TARGET_PHRASE_MAP.items():
            if phrase in lowered_text:
                return self._normalize_symbol(target)
        return self._normalize_symbol(raw_target)

    def _normalize_symbol(self, symbol: str) -> str:
        normalized = symbol.strip().lower().replace("_", " ")

        for canonical, aliases in SYMBOL_ALIASES.items():
            if normalized in aliases:
                return canonical

        return normalized

    def _augment_grade_6_7_triggers(self, text: str, triggers: list[str]) -> list[str]:
        seen = set(triggers)
        for phrase in GRADE_6_7_TRIGGER_PHRASES:
            if phrase in text.lower() and phrase not in seen:
                triggers.append(phrase)
                seen.add(phrase)
        return triggers

    def _normalize_unit(self, unit: str) -> str:
        normalized = unit.lower().replace("seconds", "s").replace("second", "s").replace("sec", "s")
        normalized = normalized.replace("meters", "m").replace("meter", "m").replace("metres", "m").replace("metre", "m")
        return normalized.replace("m/s2", "m/s^2")
