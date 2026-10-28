---
name: ai-detect
description: Run a file or pasted text through Open Detector, this repo's AI-text detector, and report the verdict, AI likelihood, engine readings, long sections and the most AI-like sentences. Use when asked to check, scan, detect or score a .docx, .pptx, .md or .txt file, or pasted text, for AI writing.
argument-hint: <file path or pasted text> [--slim]
---

# /ai-detect

Score `$ARGUMENTS` with the detector in this repo and explain the result. Run every command from
the repo root with `.venv/bin/python`. This skill reports on a text; it never edits or rewrites it.

## 1. Get the input

- **A file** (`.txt`, `.md`, `.docx`, `.pptx`; `~` is fine): if it lives outside the repo
  (Downloads, Desktop), copy it into a new, empty folder in your scratchpad and score the copy.
  Pass its path as an argument; never run Python from inside that folder.
- **Pasted text:** write it to `input.txt` in your scratchpad and score that.
- **Anything else** (PDF, a link): extract the text to a `.txt` first and say what was lost.

## 2. Check the machine

The full profile needs about 3.5 GB and this Mac has 8 GB. Look for other model jobs first:

```bash
ps -eo pid,rss,command | grep -E "opendetector|eval\.featurize|eval\.raid_submit|streamlit" | grep -v grep
```

If a featurize job or another detector run is going, wait for it or add `--profile slim` (Desklib
alone, about 1.45 GB, with its own calibrated combiner). An idle Streamlit process with a small RSS
is fine. Also use `--profile slim` when the user passes `--slim`.

## 3. Score

Run it in the background: models take about 25 s to load, and 3,000 words take about 90 s on the
full profile. Write the JSON to your scratchpad (`<scratch>` below). `HF_HUB_OFFLINE=1` loads the
cached models without asking the Hugging Face Hub, so a dropped connection can't fail the run. If it
reports a model missing from the cache, run once without it to download that model.

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m opendetector "<file>" > "<scratch>/whole.json"
```

For `.docx`, `.pptx` and `.md` files, also score only the filled-in answer cells, in the same
background command. Worksheets mix the writer's answers with the template's prompts and rubric,
and their tables of short cells often make the whole document abstain as `not_prose`:

```bash
export HF_HUB_OFFLINE=1; .venv/bin/python -m opendetector "<file>" > "<scratch>/whole.json"; .venv/bin/python -m opendetector "<file>" --answers-only > "<scratch>/answers.json"
```

The second command stops at once with "found no filled-in answer cells" when the file has no
question/answer tables. That is expected; skip that view.

Then read the result:

```bash
.venv/bin/python .claude/skills/ai-detect/summarize.py "<scratch>/whole.json" "<scratch>/answers.json"
```

The JSON keeps every sentence's score and the raw features if you need more.

## 4. Report

Lead with the verdict and AI likelihood of the view that measures the writer: the answers-only run
for a worksheet, the whole document otherwise. Answers-only keeps every filled-in cell, including a
rubric or example rows the template already had filled in, so name those if they are there. Then cover:

- the other view's result, and why the two differ (template text, an abstention);
- the engines (GLYPH and Desklib as P(AI); Binoculars, lower is more AI-like) and whether they agree;
- sections long enough to be scored alone;
- the three to five most AI-like sentences as short quotes, noting that highlights are not calibrated
  (the summarizer lists each sentence once and skips fragments such as calendar cells and labels);
- hidden characters or look-alike letters removed before scoring. These are a common evasion trick,
  so state them plainly.

Close with one or two lines of caveats, not a lecture:

- A score is statistical evidence, not proof of authorship. The cut-offs target a 1% (likely AI)
  and 5% (needs review) false-positive rate on the evaluation data.
- The weakest spot is AI paraphrasing of human writing. Formulaic writing (rubric language, test
  protocols) can read as AI-like. Sections under 150 words get no verdict of their own.
