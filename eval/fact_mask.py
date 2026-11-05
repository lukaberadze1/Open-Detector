"""Names and numbers held out of a rewrite: a model sees `[F1]` where the text says "Themistocles", with a legend saying
what each one stands for, and writes the prose around them. Told only to keep them, models dropped them: on
eval/reports/revise.md a revision kept 55% of a document's names and numbers, and the checks that catch it can only
throw the rewrite away. Citations, URLs, math and code are held the same way (after StealthHumanizer's
lib/protect-regions.ts and lib/academic-inversion.ts, MIT; names and numbers are this project's addition).

A sentence's first word is a name only when the document also capitalizes it mid-sentence, since "However" looks like
"Salamis" there. A number written out in words stays as it is. Pure Python, shipped with the Red team page.
"""

from __future__ import annotations

import re

from opendetector.segment import split_sentences

_TOKEN = re.compile(r"\\?\[\s*[Ff]\s*(\d+)\s*\\?\]")
_ARTICLE = re.compile(r"\b([Aa])n?(\s+)(?=\\?\[\s*[Ff]\s*\d+\s*\\?\])")
_HELD_WHOLE = (
    re.compile(r"```.+?```|`[^`\n]+`", re.S),  # code
    re.compile(r"\$\$.+?\$\$|\\\[.+?\\\]|\\\(.+?\\\)|\$(?![\s\d])[^$\n]+?(?<!\s)\$", re.S),  # math, not "$5 and $6"
    re.compile(r"https?://[^\s<>)\]\"']+?(?=[.,!?;:]?(?:[\s<>)\]\"']|$))"),  # a URL, not the full stop after it
    re.compile(r"[\w.%+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}"),  # an email address
    re.compile(r"\[\d+(?:\s*[-,\u2013]\s*\d+)*\]"),  # [1], [1-3]
    re.compile(r"\([A-Z][\w'\u2019-]+(?:\s+et al\.?|\s+(?:and|&)\s+[A-Z][\w'\u2019-]+)?,\s*\d{4}[a-z]?"
               r"(?:;[^)]*)?\)"),  # (Smith, 2020)
)
_NUMBER = re.compile(r"(?<![\w.,\[])(?:[$\u20ac\u00a3]\s?)?\d(?:[\d,.]*\d)?(?:st|nd|rd|th|s)?"
                     r"(?:\s?%|\s(?:million|billion|trillion|thousand)\b|\s(?:BCE|BC|AD|CE)\b"
                     r"|(?:km|kg|cm|mm|mph|GB|MB|kHz|Hz)\b)?(?![\w])")
_NAME_WORD = r"(?:[A-Z][a-z]+(?:-[A-Z]?[a-z]+)*|[A-Z]{2,}\d*)"
_NAME = re.compile(rf"(?<![\w'\u2019-]){_NAME_WORD}(?:\s+(?:of\s+(?:the\s+)?)?{_NAME_WORD})*(?![\w-])")
_OPENS_AFTER = ".!?:\"\u201c\u2018\n"
_CONSONANT_SOUND = ("uni", "use", "usa", "usu", "uti", "eu", "one", "once")


def _sentence_openings(text: str) -> set[int]:
    """Where in `text` a capital says nothing about a name: at a sentence's start, or after a colon or a quote mark."""
    starts = {sentence.start for sentence in split_sentences(text)}
    return starts | {match.end() for match in re.finditer(rf"[{_OPENS_AFTER}]\s*", text)} | {0}


def _names_seen(text: str) -> set[str]:
    """The words `text` capitalizes where a capital can only mean a name."""
    openings = _sentence_openings(text)
    names = set()
    for match in _NAME.finditer(text):
        words = match.group().split()
        names.update(words[1:] if match.start() in openings else words)
    return names


def article_for(value: str) -> str:
    """'a' or 'an' before a held name or number, by how it is said."""
    low = value.lstrip("$\u20ac\u00a3 ").lower()
    if value[:1].isdigit():
        vowel = low.startswith("8") or low.split(",")[0].split(".")[0] in {"11", "18"}
    elif value.isupper():
        vowel = low[:1] in set("aefhilmnorsx")
    else:
        vowel = low[:1] in set("aeiou") and not low.startswith(_CONSONANT_SOUND)
    return "an" if vowel else "a"


class FactMask:
    """What a group of passages hides: each distinct value has one placeholder, in every passage it is in."""

    def __init__(self, *texts: str):
        self.values: list[str] = []
        self._known = set().union(*(_names_seen(text) for text in texts)) if texts else set()

    def _token(self, value: str) -> str:
        if value not in self.values:
            self.values.append(value)
        return f"[F{self.values.index(value) + 1}]"

    def hide(self, text: str) -> str:
        """`text` with its names, numbers, citations, URLs, math and code as placeholders."""
        text = _TOKEN.sub(lambda match: self._token(match.group()), text)  # a source's own "[F1]" is held too
        for pattern in _HELD_WHOLE:
            text = pattern.sub(lambda match: self._token(match.group()), text)
        text = _NUMBER.sub(lambda match: self._token(match.group()), text)
        openings = _sentence_openings(text)

        def name(match: re.Match) -> str:
            found = match.group()
            first = found.split()[0]
            if match.start() not in openings or first in self._known:
                return self._token(found)
            rest = found[len(first):]
            return first + rest[:len(rest) - len(rest.lstrip())] + self._token(rest.strip()) if rest.strip() else found

        return _NAME.sub(name, text)

    def _value(self, number: str) -> str | None:
        return self.values[int(number) - 1] if 1 <= int(number) <= len(self.values) else None

    def restore(self, text: str) -> str:
        """`text` with its placeholders back as what they stand for, and an article before one made to fit it. A
        placeholder that stands for nothing stays, which `lost` reports."""
        def article(match: re.Match) -> str:
            found = _TOKEN.match(text, match.end())
            value = self._value(found.group(1)) if found else None
            if value is None:
                return match.group()
            word = article_for(value)
            return (word.capitalize() if match.group(1).isupper() else word) + match.group(2)

        fitted = _ARTICLE.sub(article, text)
        return _TOKEN.sub(lambda match: self._value(match.group(1)) or match.group(), fitted)

    def lost(self, hidden: str, reply: str) -> list[str]:
        """What a reply to the passage `hidden` got wrong: each placeholder it left out, unless it wrote out what that
        stands for instead, and each one it made up."""
        restored = self.restore(reply)
        held = dict.fromkeys(_TOKEN.findall(hidden))
        gone = [self.values[int(number) - 1] for number in held if self.values[int(number) - 1] not in restored]
        return gone + [match.group() for match in _TOKEN.finditer(reply) if self._value(match.group(1)) is None]

    def legend(self, hidden: list[str]) -> str:
        """The prompt's note on the placeholders in these passages, or nothing when they have none."""
        numbers = sorted({int(number) for text in hidden for number in _TOKEN.findall(text)})
        if not numbers:
            return ""
        lines = "\n".join(f"[F{number}] = {self.values[number - 1]}" for number in numbers)
        return ("\n\nThe blocks use placeholders such as [F1] for names, numbers and references that have to stay "
                "exactly as they are. Use every placeholder a block has, written just as it is, wherever your "
                "sentences need it, and make up none. They stand for:\n" + lines)


LOST = ("Your last answer left placeholders out of some blocks or made up new ones. Write those blocks again, using "
        "every placeholder each one has.")
