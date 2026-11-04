"""Deterministic rule transformations for labeled, offline adversarial evaluation data.

The engine has no model, random sampling, or network access. Rules are loaded from JSON, and
each rule is rejected unless it preserves the numeric and negation tokens it touches. Protected
math, citations, and fenced code are restored byte-for-byte after the prose pass.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_RULES_DIR = Path(__file__).with_name("rules")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*", re.UNICODE)
_WORD = re.compile(r"[\w\u2019'-]+", re.UNICODE)
_NEGATIONS = {"no", "not", "never", "neither", "nor", "none", "nothing", "without", "cannot",
              "can't", "couldn't", "didn't", "doesn't", "don't", "hadn't", "hasn't", "haven't",
              "isn't", "mustn't", "shouldn't", "wasn't", "weren't", "won't", "wouldn't"}
_PROTECTED = (
    ("CODE_FENCE", re.compile(r"```[^\n]*\n.*?```|```[^`\n]*```", re.DOTALL)),
    ("MATH_BLOCK", re.compile(r"\\begin\{(?P<env>equation\*?|align\*?|gather\*?|multline\*?)\}"
                              r".*?\\end\{(?P=env)\}", re.DOTALL)),
    ("MATH_DISPLAY", re.compile(r"\$\$(?:\\.|[^$\\])+\$\$", re.DOTALL)),
    ("MATH_INLINE", re.compile(r"(?<!\$)\$(?:\\.|[^$\\])+\$(?!\$)|"
                                r"\\\((?:\\.|[^\\])+?\\\)|\\\[(?:\\.|[^\\])+?\\\]", re.DOTALL)),
    ("CITATION", re.compile(r"\\(?:cite[a-z]*|parencite|textcite|autocite|eqref|autoref|ref)"
                            r"(?:\[[^\]]*\]){0,2}\{[^}]*\}")),
)


@dataclass(frozen=True)
class NgramRule:
    source: str
    replacement: str
    category: str = "general"
    domains: tuple[str, ...] = ()


class DeterministicEngine:
    """Load vetted JSON rules and apply repeatable, fact-gated phrase substitutions."""

    def __init__(self, rules_dir: str | Path | None = None) -> None:
        self.rules_path = Path(rules_dir) if rules_dir is not None else DEFAULT_RULES_DIR
        self.ngram_rules: list[NgramRule] = []
        self.pdtb_rules: dict[str, tuple[str, ...]] = {}
        self._load_decoupled_rules()

    def _load_decoupled_rules(self) -> None:
        """Load the versioned rules format, with support for the earlier list-of-maps format."""
        ngram_file = self.rules_path / "ngrams.json"
        if ngram_file.exists():
            try:
                data = json.loads(ngram_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Could not read n-gram rules from {ngram_file}.") from exc
            self.ngram_rules = self._parse_ngram_rules(data)

        pdtb_file = self.rules_path / "pdtb_markers.json"
        if pdtb_file.exists():
            try:
                data = json.loads(pdtb_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Could not read PDTB rules from {pdtb_file}.") from exc
            self.pdtb_rules = self._parse_pdtb_rules(data)

    @staticmethod
    def _parse_ngram_rules(data: Any) -> list[NgramRule]:
        if isinstance(data, dict):
            entries = data.get("rules", [])
        elif isinstance(data, list):
            entries = data
        else:
            raise ValueError("ngrams.json must be an object with a 'rules' list or a list of rule objects.")
        rules: list[NgramRule] = []
        for group in entries:
            if not isinstance(group, dict) or "__prompt__" in group:
                continue
            if "source" in group or "replacement" in group:
                source, replacement = group.get("source"), group.get("replacement")
                category = group.get("category", "general")
                domains = group.get("domains", [])
                rules.append(DeterministicEngine._make_rule(source, replacement, category, domains))
            else:
                # Legacy map groups: {"general": {"phrase": "replacement"}}.
                for category, mappings in group.items():
                    if category == "__prompt__":
                        continue
                    if not isinstance(mappings, dict):
                        raise ValueError(f"N-gram rule group {category!r} must be a mapping.")
                    for source, replacement in mappings.items():
                        rules.append(DeterministicEngine._make_rule(source, replacement, category, [category]))
        seen = set()
        for rule in rules:
            key = (rule.source, rule.category)
            if key in seen:
                raise ValueError(f"Duplicate n-gram rule {rule.source!r} in category {rule.category!r}.")
            seen.add(key)
        return sorted(rules, key=lambda rule: (-len(rule.source), rule.source, rule.category))

    @staticmethod
    def _make_rule(source: Any, replacement: Any, category: Any, domains: Any) -> NgramRule:
        if not isinstance(source, str) or not source.strip() or not isinstance(replacement, str) or \
                not replacement.strip():
            raise ValueError("Every n-gram rule needs non-empty string 'source' and 'replacement' fields.")
        if source != source.strip() or replacement != replacement.strip():
            raise ValueError("N-gram rule phrases may not have leading or trailing whitespace.")
        if source != source.casefold() or replacement != replacement.casefold():
            raise ValueError("N-gram rule phrases must be lowercase so capitalized names remain untouched.")
        if not isinstance(category, str) or not category.strip():
            raise ValueError("N-gram rule category must be a non-empty string.")
        if not isinstance(domains, list) or any(not isinstance(item, str) for item in domains):
            raise ValueError(f"Domains for n-gram rule {source!r} must be a list of strings.")
        if source == replacement:
            raise ValueError(f"N-gram rule {source!r} has no effect.")
        if DeterministicEngine._invariants(source) != DeterministicEngine._invariants(replacement):
            raise ValueError(f"N-gram rule {source!r} would change a number or negation token.")
        return NgramRule(source, replacement, category, tuple(domains))

    @staticmethod
    def _parse_pdtb_rules(data: Any) -> dict[str, tuple[str, ...]]:
        groups = data.get("relations", data) if isinstance(data, dict) else None
        if not isinstance(groups, dict):
            raise ValueError("pdtb_markers.json must map relation names to marker lists.")
        result = {}
        all_markers = set()
        for relation, markers in groups.items():
            if relation == "version":
                continue
            if not isinstance(relation, str) or not relation.strip() or not isinstance(markers, list) or \
                    len(markers) < 2 or any(not isinstance(marker, str) or not marker.strip() for marker in markers):
                raise ValueError(f"PDTB relation {relation!r} needs at least two non-empty marker strings.")
            normalized = tuple(markers)
            if any(marker != marker.casefold() for marker in normalized):
                raise ValueError(f"PDTB markers in {relation!r} must be lowercase so names remain untouched.")
            if len(set(normalized)) != len(normalized):
                raise ValueError(f"PDTB relation {relation!r} contains duplicate markers.")
            if any(DeterministicEngine._invariants(marker) != DeterministicEngine._invariants(markers[0])
                   for marker in markers):
                raise ValueError(f"PDTB relation {relation!r} mixes markers with different negation tokens.")
            overlap = all_markers.intersection(normalized)
            if overlap:
                raise ValueError(f"PDTB markers must belong to one relation group; duplicates: {sorted(overlap)}")
            all_markers.update(normalized)
            result[relation] = normalized
        return result

    @staticmethod
    def _invariants(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        numbers = tuple(_NUMBER.findall(text))
        negations = tuple(word.casefold() for word in _WORD.findall(text) if word.casefold() in _NEGATIONS)
        return numbers, negations

    def protect_ast_latex(self, text: str) -> tuple[str, dict[str, str]]:
        """Replace math, citations, and fenced code with opaque UUID tokens."""
        token_map: dict[str, str] = {}
        token_index = 0
        for label, pattern in _PROTECTED:
            def replace(match: re.Match[str], kind: str = label, current_text: str = text) -> str:
                nonlocal token_index
                suffix = uuid.uuid5(uuid.NAMESPACE_URL, f"opendetector:{token_index}:{match.group(0)}").hex
                token = f"__OD_PROTECTED_{kind}_{suffix}__"
                token_index += 1
                while token in current_text or token in token_map:
                    suffix = uuid.uuid5(uuid.NAMESPACE_URL, f"opendetector:{token_index}:{match.group(0)}").hex
                    token = f"__OD_PROTECTED_{kind}_{suffix}__"
                    token_index += 1
                token_map[token] = match.group(0)
                return token

            text = pattern.sub(replace, text)
        return text, token_map

    @staticmethod
    def restore_ast_latex(text: str, token_map: dict[str, str]) -> str:
        """Restore each protected span exactly once, rejecting missing or duplicated tokens."""
        for token in token_map:
            if text.count(token) != 1:
                raise ValueError("A protected math, citation, or code token was changed or duplicated.")
        for token, original in reversed(tuple(token_map.items())):
            text = text.replace(token, original)
        return text

    def disrupt_ngram_sequences(self, text: str, domain: str | None = None) -> str:
        """Apply longest-first, non-overlapping n-gram rules without changing factual invariants."""
        protected, token_map = self.protect_ast_latex(text)
        rules = [rule for rule in self.ngram_rules if domain is None or not rule.domains or domain in rule.domains]
        if rules:
            alternatives = "|".join(re.escape(rule.source) for rule in rules)
            pattern = re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)")
            by_source = {rule.source: rule.replacement for rule in rules}
            protected = pattern.sub(lambda match: by_source[match.group(0)], protected)
        try:
            restored = self.restore_ast_latex(protected, token_map)
        except ValueError:
            return text
        return restored if self._invariants(text) == self._invariants(restored) else text

    def rotate_pdtb_markers(self, text: str) -> str:
        """Rotate recognized discourse markers to the next marker in their same relation group."""
        protected, token_map = self.protect_ast_latex(text)
        for markers in self.pdtb_rules.values():
            pattern = re.compile(rf"(?<!\w)(?:{'|'.join(re.escape(item) for item in markers)})(?!\w)")
            rotation = {marker: markers[(index + 1) % len(markers)] for index, marker in enumerate(markers)}
            protected = pattern.sub(lambda match, mapping=rotation: mapping[match.group(0)], protected)
        try:
            restored = self.restore_ast_latex(protected, token_map)
        except ValueError:
            return text
        return restored if self._invariants(text) == self._invariants(restored) else text

    def transform(self, text: str, domain: str | None = None) -> str:
        """Run the deterministic n-gram pass followed by the discourse-marker rotation."""
        intermediate = self.disrupt_ngram_sequences(text, domain=domain)
        return self.rotate_pdtb_markers(intermediate)
