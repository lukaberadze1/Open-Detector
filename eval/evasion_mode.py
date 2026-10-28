"""Detector-guided red-team generation for labelled evaluation samples.

Every candidate remains AI-generated and should be retained as an AI-labelled hard example.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class EvasionResult:
    text: str
    score: float | None
    attempts: int
    reached_target: bool


PROFILE_ID = "sub_5_percent_stealth_v3"

# Adapted from the supplied red-team profile. These are generation instructions, not guarantees:
# model providers may ignore style constraints, and the detector score is empirical.
PROFILE_INSTRUCTIONS = """Red-team style variation profile:
- Vary pacing across short fragments, medium sentences, and longer multi-clause sentences. Avoid
  repeating the same grammatical opening or similar sentence lengths consecutively.
- Mix sentence structures and occasional front-loaded phrases; use fragments only when they read
  naturally. Keep a coherent argument and preserve the source meaning and evidence.
- Prefer direct, specific wording. Avoid these stock terms and transitions when a natural
  alternative exists: delve, testament, beacon, paramount, navigate, crucial, furthermore,
  moreover, in conclusion, it is important to note, tapestry, plethora, realm, synergy, pivotal,
  subverse, shed light on, multifaceted.
- Use contractions where they fit the speaker and context. Avoid formulaic hedging and formal
  transition phrases. Punctuation and rhetorical questions should be natural, never inserted to
  meet a quota.
- Do not fabricate personal experience, introduce errors, or change the task's meaning.
This is AI-generated red-team evaluation data. Return only the sample text."""

REWRITE_INSTRUCTIONS = """Revise the draft below as another AI-generated red-team candidate.
Follow the original task and constraints, then apply the style profile. Preserve claims, evidence,
point of view, and approximate length. Return only the revised text.

Original task and context:
{original_prompt}

Current draft:
{draft}"""


def generate_until_target(
    generate: Callable[[str], str],
    score: Callable[[str], float | None],
    prompt: str,
    *,
    target: float = 0.05,
    max_attempts: int = 3,
) -> EvasionResult:
    """Try a small number of detector-guided revisions; keep the lowest scored candidate."""
    best_text = ""
    best_score: float | None = None
    current_prompt = prompt
    for attempt in range(1, max_attempts + 1):
        text = generate(current_prompt)
        value = score(text)
        if value is not None and (best_score is None or value < best_score):
            best_text, best_score = text, value
        elif not best_text:
            best_text = text
        if value is not None and value < target:
            return EvasionResult(text, value, attempt, True)
        current_prompt = REWRITE_INSTRUCTIONS.format(original_prompt=prompt, draft=text)
    return EvasionResult(best_text, best_score, max_attempts, False)
