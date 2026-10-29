---
name: style-profile
description: Weekly update of the living style profile (data/styles/alexandria_profile.json) from the writer's new notes in The Alexandria Archive, their Obsidian vault. Measures the new writing, stacks it onto everything measured before, and revises the profile in place in its exact JSON format without dropping older patterns. Use when asked to update, refresh or rebuild the style profile, or when the weekly scheduled task runs.
---

# /style-profile

The profile is a JSON file in a fixed format (the one in `data/styles/pasted_style_profile.json`). It
describes measured writing patterns, and `eval/generate_essays.py --style-profile` uses it as a
controlled condition to produce labelled AI challenge samples for adversarial detector evaluation.
Each week it absorbs the writer's
new notes. **It stacks: what was learned in earlier weeks stays unless the evidence turns against
it.** Run every command from the repo root.

## 1. Measure

```bash
.venv/bin/python update_style_profile.py
```

This reads the vault (iCloud, `The Alexandria Archive`), adds every paragraph not seen before to
`data/styles/alexandria/corpus.jsonl` dated this ISO week, copies the profile to
`data/styles/alexandria/history/alexandria_profile_<week>.json` (the state before this week's edit),
and writes:

- `data/styles/alexandria/measurements.json`
  - `profile_numbers`: the values for `avg_sentence_length_words`, `min_max_sentence_length` and
    `sentence_length_distribution`, already blended with the starting profile.
  - `stacked`: all the archive's writing, recent weeks weighted more (12-week half-life, never below 0.25).
  - `this_week`: the new writing alone, and `drift`, which compares the two.
  - `all_time_unweighted`: raw counts, for claims like "1 in about 36,000 words".
  - `weeks`: one row per week, with its weight.
  - `skipped_as_pasted`: lines left out as pasted or vault-generated.
- `data/styles/alexandria/new_this_week.md`: the new paragraphs themselves.

If the vault is missing or notes are listed as unreadable (iCloud has not downloaded them), say so
in the report and carry on with what was read.

## 2. Read

Read `measurements.json` and `new_this_week.md` in full, then the current profile. The new
paragraphs are the evidence. Read them for habits the numbers can't see: how sentences start and
join, fragments, run-ons, asides in parentheses, how the writer argues, register, recurring slips,
stray capitals, words they reach for.

Only the writer's own words count. The script already drops callouts, quotes, pasted lines and the
vault's review notes. If a paragraph still reads as pasted (encyclopedic, polished, no slips at
all, in a style the writer never uses), leave it out of the evidence and mention it in the changelog.

## 3. Edit `data/styles/alexandria_profile.json`

### The format is fixed
Same keys, in the same order, with the same value types. Never add, rename or remove a key. All new
evidence goes inside the existing strings and lists.

### The numbers
- Copy `avg_sentence_length_words`, `min_max_sentence_length` and `sentence_length_distribution`
  from `profile_numbers` exactly.
- Figures inside the prose (coefficient of variation, rates per 1,000 words, percentages) describe
  registers. Keep the figures for essays, fiction, reports and personal writing: the vault holds
  only history notes, so it can't contradict them. Add or update figures for **history notes
  (Alexandria Archive)** from `stacked`, with the date range they cover.

### Stacking: how patterns live and die
- **Keep.** Never delete a pattern, example, signature word, banned phrase or rule because it was
  missing this week. One quiet week of history notes is not evidence against an essay habit.
- **Add.** A habit that is new this week (a slip, an opener, a favourite word, a structure) goes into
  the field it belongs to, tagged with the week: `'atribbuted' (new 2026-W45)`. Give a short real
  example from `new_this_week.md` (under 12 words), never a long quote.
- **Confirm.** When a tagged item shows up again in a later week, change its tag to `(since 2026-W45)`.
- **Revise.** Change what an item claims only when `stacked` contradicts it with at least 1,000
  words over at least 3 different weeks. Then say what changed instead of erasing it:
  `'However' takes no comma (0 of 49 in essays; in notes since 2026-W46, 3 of 10 do)`.
- **Fade, never forget.** An item not seen for 12 weeks or more gets `(not seen since 2026-W45)`
  and stays.
- **Lists only grow**: `signature_vocabulary`, `banned_ai_cliches` and both constraint lists.
  - The one exception is `banned_ai_cliches`. When `banned_phrases_used` shows the writer uses a
    banned phrase in their own sentences, in at least 3 different weeks, move it to
    `signature_vocabulary` with the register and the count. Banning a word the writer really uses
    would make the imitation less like them.
  - Until then, keep the ban and record the use in the matching prose field. For example,
    `forbidden_structures` or `transition_mechanics` can say "history notes do use 'It is
    important to note' (7 in 36,000 words)".
- **Rules** in `negative_constraints` and `positive_constraints` follow the same evidence. Update
  their numbers, and add a new rule only for a pattern seen in at least 2 different weeks.
  `golden_transformation_rule` stays one sentence; revise it only when the overall character has
  changed.

### Write like the profile
Write plain, specific, measured statements in the profile's existing register: numbers, then a
short real example. Name the register each observation comes from (essays, notes, fiction).

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
- Numbers: mean sentence 18.6 → 18.7; under 6 words 6.6% → 6.5%.
- Added: ...
- Confirmed: ...
- Revised: ... (old → new, with the evidence)
- Fading: ...
- Ignored as pasted: ...
```

With no new writing, update only the numbers from `profile_numbers`; they move a little as the
weights age. Log "No new writing" with the numbers that changed.

## 6. Report

In 3–5 lines: how much new writing there was, the biggest change to the profile, anything left
out, and the snapshot path to undo this week's edit.
