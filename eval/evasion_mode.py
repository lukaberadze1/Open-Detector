"""Adaptive detector-challenge generation for labelled evaluation samples.

Every candidate remains AI-generated and should be retained as an AI-labelled hard example.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable


@dataclass
class EvasionResult:
    text: str
    score: float | None
    attempts: int
    reached_target: bool
    style_score: float | None = None


PROFILE_ID = "adaptive_style_challenge_v1"

# This is an explicit style-conditioned benchmark condition, not a claim of human authorship.
# Providers may ignore style constraints, and detector scores are empirical and build-specific.
PROFILE_INSTRUCTIONS = """Adversarial style-profile challenge condition:
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
This is synthetic, AI-generated adversarial evaluation data. Do not claim it was written by a human.
Return only the sample text."""

REWRITE_INSTRUCTIONS = """Revise the draft below as another AI-generated adversarial challenge candidate.
Follow the original task and constraints, then apply the style profile. Preserve claims, evidence,
point of view, and approximate length. Use varied sentence lengths and openings, specific wording,
and the profile's usual punctuation and vocabulary. Do not add errors or unsupported personal detail.
Return only the revised text.

Original task and context:
{original_prompt}

Current draft:
{draft}"""


def generate_until_target(
    generate: Callable[[str], str],
    score: Callable[[str], float | dict | None],
    prompt: str,
    *,
    target: float = 0.05,
    max_attempts: int = 3,
    style_floor: float = 35.0,
) -> EvasionResult:
    """Try candidates and balance detector likelihood with measured style fit.

    When style is measurable, candidates above the floor compete on the mean of normalized
    detector and style fit. Below it, style fit remains a priority until a candidate clears it.
    Legacy float callbacks remain supported.
    """
    best_text = ""
    best_score: float | None = None
    best_style: float | None = None
    best_rank: tuple[float, ...] | None = None
    current_prompt = prompt
    for _ in range(max_attempts):
        text = generate(current_prompt)
        measured = score(text)
        if isinstance(measured, dict):
            value = measured.get("ai_probability")
            style = measured.get("style_match")
        else:
            value, style = measured, None
        rank = _candidate_rank(value, style, style_floor)
        if best_rank is None or rank > best_rank:
            best_text, best_score, best_style, best_rank = text, value, style, rank
        elif not best_text:
            best_text = text
        current_prompt = REWRITE_INSTRUCTIONS.format(original_prompt=prompt, draft=text)
        if value is not None or style is not None:
            current_prompt += (f"\n\nCandidate assessment: detector likelihood "
                               f"{value if value is not None else 'unavailable'}; "
                               f"style match {style if style is not None else 'unavailable'}%. Improve whichever "
                               "measure is weaker while keeping the same meaning and length.")
        if isinstance(measured, dict) and measured.get("style_feedback"):
            current_prompt += "\n\nSpecific style adjustments:\n- " + "\n- ".join(measured["style_feedback"])
    reached = (best_score is not None and best_score < target and
               (best_style is None or best_style >= style_floor))
    return EvasionResult(best_text, best_score, max_attempts, reached, best_style)


def _candidate_rank(ai: float | None, style: float | None, style_floor: float) -> tuple[float, ...]:
    """Prefer candidates meeting the style floor; then balance both measures on 0-1 scales."""
    detector_fit = 0.5 if ai is None else 1.0 - max(0.0, min(1.0, float(ai)))
    style_fit = 0.5 if style is None else max(0.0, min(100.0, float(style))) / 100.0
    meets_floor = style is not None and style >= style_floor
    if style is None:
        return (0.0, detector_fit)
    if meets_floor:
        return (1.0, (detector_fit + style_fit) / 2.0, detector_fit)
    return (0.0, (detector_fit + style_fit) / 2.0, style_fit)
