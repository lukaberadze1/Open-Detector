---
name: style-profile
description: Weekly update of the living style profile (data/styles/alexandria_profile.json) from the writer's new notes in The Alexandria Archive, their Obsidian vault. Measures the new writing, stacks it onto everything measured before, and revises the profile in place in its exact JSON format without dropping older patterns. Use when asked to update, refresh or rebuild the style profile, or when the weekly scheduled task runs.
---

# /style-profile

The profile is a JSON file in a fixed format, the one in `style_profile_template.json` ("registers-v2").
It describes the writer's measured writing patterns, and the Red team page, `eval/generate_essays.py
--style-profile` and `eval/tune_generation.py` send it to the model so generated samples read like the
writer. Each week it absorbs the writer's new notes. **It stacks: what was learned in earlier weeks
stays unless the evidence turns against it.** Run every command from the repo root.

## The two registers

The writer writes differently when they finish a piece for someone else and when they take quick notes
for themselves. The profile keeps the two apart, and the generator sends only one of them:

- **finalized**: essays, reports, stories, answers and reflections. Its numbers come from the starting
  profile (about 54,000 words of finished pieces), blended with:
  - the finished pieces in `data/styles/alexandria/finalized/`: `.md`, `.txt`, `.docx`, `.pptx` or
    `.pdf`, each counted in full unless `finalized/weights.json` gives it less (`{"report.pdf": 0.9}`).
    Only the writer's own prose counts: long quotations, verse, equations, tables and the questions of
    a question/answer table are left out, and so is any text a blank template saved in
    `finalized/templates/` also has. A paragraph repeated across files counts once.
  - the vault's notes by the rung the writer gave them in their tags: **Gold** (finished) at 0.5,
    **Silver** (drafted) at 0.2, Bronze and fleeting captures not at all (`LADDER`). A finished note is
    still a note, so it never counts as fully finished work.
- **notes**: the vault. Everything the script reads from Obsidian is notes, however polished it
  looks, Gold notes included. Lesson captures typed while listening are the fastest, loosest kind.

**A habit seen in notes never goes into finalized** unless finished pieces show it too. Quick-note
slips, lowercase lines, run-ons and terms written down as heard are how the writer writes fast, not
how they write finished work. Silver and Gold notes move the finalized numbers a little, but a habit
needs a finished piece to show it before it goes into the finalized prose.

## 1. Measure

```bash
.venv/bin/python update_style_profile.py
```

This reads the vault (iCloud, `The Alexandria Archive`), adds every paragraph not seen before to
`data/styles/alexandria/corpus.jsonl` dated this ISO week, copies the profile to
`data/styles/alexandria/history/alexandria_profile_<week>.json` (the state before this week's first
edit), and writes:

- `data/styles/alexandria/measurements.json`
  - `register_numbers`: the rhythm numbers for each register, in the profile's fields. `notes` comes
    from the vault, recent weeks weighted more; `finalized` is the starting profile with any finished
    pieces blended in.
  - `stacked`: all the vault's writing, recent weeks weighted more (12-week half-life, never below 0.25).
  - `this_week`: the new writing alone, and `drift`, which compares the two.
  - `all_time_unweighted`: raw counts, including `most_common_unknown` (the misspelling counts).
  - `weeks`: one row per week, with its weight.
  - `finished_pieces`: what went into the finalized numbers: each file with its weight and words, and
    the Silver and Gold notes with theirs.
  - `finalized_own`: those finished pieces and rated notes measured like `stacked`, at their weights:
    the figures for the finalized prose.
  - `skipped_as_pasted`: lines left out as pasted or vault-generated.
- `data/styles/alexandria/new_this_week.md`: the new paragraphs themselves.

If the vault is missing or notes are listed as unreadable (iCloud has not downloaded them), say so in
the report and carry on with what was read. If you edit the profile twice in one week, copy it to
`history/alexandria_profile_<week>_before-<date>.json` first, since the week's snapshot is from
before its first edit.

## 2. Read

Read `measurements.json` and `new_this_week.md` in full, then the current profile. The new paragraphs
are the evidence. Read them for habits the numbers can't see: how sentences start and join,
fragments, run-ons, asides in parentheses, how the writer argues, register, recurring slips, stray
capitals, words they reach for.

Only the writer's own words count. The script already drops callouts, quotes, pasted lines and the
vault's review notes. If a paragraph still reads as pasted (encyclopedic, polished, no slips at all,
in a style the writer never uses), leave it out of the evidence and mention it in the changelog.

## 3. Edit `data/styles/alexandria_profile.json`

### The format is fixed
Same keys, in the same order, with the same value types, as `style_profile_template.json`. Never add,
rename or remove a key. Every entry in a register's `habits` list has exactly the template's five
fields: `habit`, `rate`, `status`, `seen`, `example`. All new evidence goes inside the existing
strings and lists.

### Frequency decides, not presence
Every habit, pattern and slip carries a **rate** (per 1,000 words, per sentence, or per eligible case)
and a **status**, worked out from the counts:

- **established**: seen in 3 or more different weeks or sources, or 10 or more times.
- **emerging**: seen in 2 weeks or sources, or 3 to 9 times.
- **once**: one note or a single occurrence. It is kept as evidence, but the generator uses it at
  most once in a long text. A one-off is never written as a rule.

Promote a habit when the counts reach the next status, and say so in the changelog. Rates come from
the measurements where they exist (`per_1000_words`, `sentence_openers_pct`, `mechanics`); for habits
the script doesn't count, count them in `new_this_week.md` and the earlier evidence and give the
count with the words it came from ("13 in 37,867 words").

### Slips by their counts
- **recurring_misspellings**: a misspelling seen 3 or more times (`most_common_unknown` in
  `all_time_unweighted`), with the word it stands for and its count. Update the counts each week.
  Leave out words that are only a proper noun in lowercase or a foreign term: those belong to `casing`
  and `vocabulary`.
- **one_off_kinds**: misspellings seen once or twice are not listed as words to reuse. Add them as
  examples of their kind (doubled letters, swapped letters, a letter or space in the wrong place,
  double capitals, homophones) with the week.
- **overall_rate**: the share of misspelled words, with how many words that means in a 300-word text.

### The numbers
- Copy `register_numbers.notes` into `registers.notes.rhythm` exactly (average, median, range,
  coefficient of variation, distribution), and `register_numbers.finalized` into
  `registers.finalized.rhythm`.
- Update the figures inside the notes prose (pronouns, vocabulary, verbs, punctuation per 1,000 words)
  from `stacked`, with the word count and date range they cover. Update finalized figures only when
  finished pieces were added, from `finalized_own`, and say which pieces they come from. Its examples
  must come from the finished pieces themselves, never from notes.

### Stacking: how patterns live and die
- **Keep.** Never delete a pattern, example, signature word, banned phrase or rule because it was
  missing this week. One quiet week of notes is not evidence against a finished-work habit.
- **Add.** A habit that is new this week goes into the register and field it belongs to, as status
  `once` (or higher if it already appears often), tagged with the week. Give a short real example
  from `new_this_week.md` (under 12 words), never a long quote.
- **Revise.** Change what an item claims only when `stacked` contradicts it with at least 1,000 words
  over at least 3 different weeks. Then say what changed instead of erasing it.
- **Fade, never forget.** An item not seen for 12 weeks or more gets `(not seen since 2026-W45)` in its
  `seen` field and stays.
- **Lists only grow**: `signature_vocabulary`, `banned_ai_cliches`, `always`, `never`,
  `recurring_misspellings` and `one_off_kinds`.
  - The one exception is `banned_ai_cliches`. When `banned_phrases_used` shows the writer uses a
    banned phrase in their own sentences, in at least 3 different weeks, move it to
    `signature_vocabulary` with the register and the count. Until then, keep the ban and record the use
    as a notes habit with its rate.
- **always** and **never** hold only what is true in both registers, with the evidence from each.
  `golden_rule` stays one sentence per register; revise it only when that register's overall character
  has changed.

### Write like the profile
Plain, specific, measured statements in the profile's existing register: numbers, then a short real
example.

## 4. Check

```bash
.venv/bin/python update_style_profile.py --check
```

It must print `format OK`. If it doesn't, fix the profile until it does. If you can't, copy this
week's snapshot from `history/` back over the profile and report the failure.

## 5. Log

Append a section to `data/styles/alexandria/CHANGELOG.md`:

```
## 2026-W45 (2026-11-08)
- New writing: 14 paragraphs, 1,240 words from 3 notes (Hannibal, tithe, Chapter 44).
- Numbers (notes): mean sentence 19.9 -> 20.1; 30+ words 16.9% -> 17.2%.
- Added: ... (register, status)
- Promoted: ... (once -> emerging, with the counts)
- Revised: ... (old -> new, with the evidence)
- Fading: ...
- Ignored as pasted: ...
```

With no new writing, update only the numbers from `register_numbers`; they move a little as the
weights age. Log "No new writing" with the numbers that changed.

## 6. Report

In 3-5 lines: how much new writing there was, the biggest change to the profile, anything left out,
and the snapshot path to undo this week's edit.
