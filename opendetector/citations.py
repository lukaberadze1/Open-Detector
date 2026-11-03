"""Citation formatting for references explicitly supplied by the user."""

from __future__ import annotations

import re


def validate_references(value: object) -> list[dict[str, str | int]]:
    if not isinstance(value, list) or len(value) > 30:
        raise ValueError("Provide a JSON list of at most 30 references.")
    result = []
    for index, ref in enumerate(value, 1):
        if not isinstance(ref, dict):
            raise ValueError(f"Reference {index} must be an object.")
        for field in ("author", "title", "year", "venue"):
            if field not in ref or not str(ref[field]).strip():
                raise ValueError(f"Reference {index} needs {field!r}.")
        try:
            year = int(ref["year"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Reference {index} needs a numeric year.") from exc
        if not 1000 <= year <= 2100:
            raise ValueError(f"Reference {index} needs a year between 1000 and 2100.")
        row = {key: str(ref[key]).strip() for key in ("author", "title", "venue")}
        if any(len(item) > 2000 for item in row.values()):
            raise ValueError(f"Reference {index} has a field longer than 2,000 characters.")
        row["year"] = year
        for optional in ("doi", "url"):
            if ref.get(optional):
                row[optional] = str(ref[optional]).strip()
                if len(row[optional]) > 2000:
                    raise ValueError(f"Reference {index} has a field longer than 2,000 characters.")
        result.append(row)
    return result


def citation_material(references: list[dict[str, str | int]]) -> tuple[str, str, str]:
    """Return prompt guidance, a formatted reference list, and matching BibTeX."""
    prompt = "\n".join(
        f"[{index}] {ref['author']} ({ref['year']}). {ref['title']}. {ref['venue']}."
        for index, ref in enumerate(references, 1)
    )
    def bib_escape(value: object) -> str:
        return str(value).replace("\\", r"\textbackslash{}").replace("{", r"\{").replace("}", r"\}")

    entries = []
    for index, ref in enumerate(references, 1):
        author_slug = re.sub(r"[^a-z0-9]+", "", str(ref["author"]).lower())[:12] or "ref"
        fields = [f"  author = {{{bib_escape(ref['author'])}}}",
                  f"  title = {{{bib_escape(ref['title'])}}}",
                  f"  year = {{{ref['year']}}}", f"  journal = {{{bib_escape(ref['venue'])}}}"]
        fields.extend(f"  {name} = {{{bib_escape(ref[name])}}}" for name in ("doi", "url") if ref.get(name))
        entries.append(f"@misc{{{author_slug}{ref['year']}_{index},\n" + ",\n".join(fields) + "\n}")
    return prompt, "References:\n" + prompt if prompt else "", "\n\n".join(entries)
