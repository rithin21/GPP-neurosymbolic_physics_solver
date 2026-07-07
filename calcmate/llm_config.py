from __future__ import annotations

"""Central DSPy language-model configuration.

CalcMate uses two distinct LLM boundaries:

* a small, cheap model for the narrow *parsing* boundaries (extraction and
  narration) -- this is the ``llama-3.1-8b-instant`` default already used by
  the extractor and narrator;
* a stronger model for the new *reasoning* boundaries (planning and the
  symbolic-solver fallback), where selection quality matters more.

Both share the same DSPy + Groq (OpenAI-compatible) transport. To avoid the two
boundaries fighting over DSPy's global ``dspy.configure(lm=...)`` state, callers
should build an LM here and use it through :func:`use_lm`, which scopes the LM
to a single call via ``dspy.context``.
"""

import os
from contextlib import contextmanager
from typing import Any, Iterator


def _groq_base() -> str:
    return os.environ.get("GROQ_API_BASE", "https://api.groq.com/openai/v1")


def build_extraction_lm() -> Any:
    """Small model for extraction/narration (parsing boundaries)."""
    import dspy

    model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
    return dspy.LM(
        f"openai/{model}",
        api_key=os.environ["GROQ_API_KEY"],
        api_base=_groq_base(),
        temperature=0.1,
        max_tokens=700,
    )


def build_reasoning_lm() -> Any:
    """Stronger model for planning + fallback (reasoning boundaries)."""
    import dspy

    # Default to a larger Groq model; override with CALCMATE_REASONING_MODEL
    # (e.g. another Groq id, or any OpenAI-compatible ``provider/model`` string).
    model = os.environ.get("CALCMATE_REASONING_MODEL", "llama-3.3-70b-versatile")
    return dspy.LM(
        f"openai/{model}",
        api_key=os.environ["GROQ_API_KEY"],
        api_base=_groq_base(),
        temperature=0.2,
        max_tokens=900,
    )


@contextmanager
def use_lm(lm: Any) -> Iterator[None]:
    """Scope ``lm`` to the enclosed DSPy calls without touching global config."""
    import dspy

    with dspy.context(lm=lm):
        yield


def reasoning_enabled() -> bool:
    """True when the LLM reasoning stages (planning + fallback) should run.

    Mirrors the narrator's guard: requires a key and an explicit opt-in flag so
    tests and offline runs stay fully deterministic.
    """
    if not os.environ.get("GROQ_API_KEY"):
        return False
    return os.environ.get("CALCMATE_USE_LLM_REASONING", "1") != "0"
