# Open Detector: ensemble detector and eval harness

```
text ─► normalize ─► sentences ─► overlapping windows ─┬─► GLYPH (DeBERTa-v3-base)   ─┐
        (zero-width,  (spans)     doc: ~350 words      ├─► Desklib (DeBERTa-v3-large) ├─► combiner ─► calibrated p ─► verdict
         homoglyphs)              local: ~100 words    └─► Binoculars (Qwen pair)     ─┘  (logistic    (thresholds set on
                                                                                         regression)  held-out human text)
```

- **Whole document, never truncated.** Text is cut into overlapping windows of whole sentences,
  and every window is scored. Sentences inherit the average score of the windows around them,
  so no sentence is ever judged alone. The windows behind the highlights stay inside paragraphs
  (very short paragraphs merge with a neighbour first), so a human paragraph next to an AI one
  isn't coloured by it.
- **Binoculars replaces raw perplexity.** Raw perplexity is the signal that flags simple prose
  and non-native writers. Binoculars normalizes it by how predictable *any* text looks to a
  second model.
- **A learned combiner replaces hand-tuned offsets.** It is a logistic regression over engine
  features, fitted on labelled data. The *Likely AI* threshold is set on held-out human texts
  for a 1% false-positive rate, and *Needs review* for 5%, using a conformal quantile that
  carries a finite-sample guarantee.
- **Evasion tricks are removed and reported.** Zero-width characters and Cyrillic/Greek
  look-alike letters are stripped before scoring, and the app says how many it found.
- **It abstains on short text and non-prose.** Under 150 words (Turnitin reportedly needs about
  300) it gives no verdict. Poetry, lists and other line-by-line formats get none either: on RAID
  this guard catches 84% of human poems and 92% of recipe-style lists, but only 0-2% of human
  prose in other domains. Turnitin doesn't score poetry, scripts or lists either.

The shipped `models/combiner.json` is fitted on a 1,400-text RAID sample. On RAID's held-out
test split, it flags 1 of 125 human documents (0.8%) as Likely AI and catches 87% of AI texts.
The full numbers are in `eval/reports/latest.md`. Without a combiner file, the app runs
**uncalibrated**: it averages the two classifiers and labels the result that way.

## Setup

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
```

```bash
.venv/bin/streamlit run app.py
```

The app has two pages, linked from the bar at the top. The sidebar picks the detector build both
pages use: **Full** (all three engines, about 3.5 GB) or **Slim** (Desklib alone on the CPU, about
1.5 GB). Machines with 8 GB of RAM or less start on Slim, which scores as well as Full on the
held-out test. Inputs and results stay put when you switch pages.

**Detect** accepts pasted text, `.md` Markdown files, `.docx` Word files and `.pptx` PowerPoint
files. The result shows the verdict and AI likelihood against the two cut-offs, each engine's
reading, sentence highlights, a sentence table and the calibration details. An uploaded document
gets a full-document score, and each extracted paragraph, table cell or text box with at least 150
words of prose is also scored on its own; shorter sections are listed without a verdict. For a
worksheet, a toggle scores just the filled-in answer cells of its question/answer tables.
Reference documents can be uploaded alongside to read; their text is never scored. The detector
does not fill templates or rewrite content to target a score.

**Red team** generates labelled AI samples to test the detector with. Enter a test prompt or upload
a Markdown, Word or PowerPoint source, then pick a task: a new sample, a rewrite of one section, a
new answer in every paired left-question/right-answer box (Markdown tables with question and answer
columns included), or a rewrite of every text block, which keeps headings, titles, questions and
tables fixed. If a file doesn't support a task, the page says why. Document tasks export a
separate Markdown/DOCX/PPTX copy that keeps the uploaded structure; the original is left unchanged.
The two batch tasks run in the background, save each block as it is written, and resume where they
stopped.

Each sample can be scored once with the detector, and its probability, verdict, calibration state
and detector build are recorded next to `label: 1` in `data/essays/local_redteam.jsonl`. These
scores help measure false negatives; the generator does not revise text based on a score. The
latest sample downloads as a `.txt` file, and it and any document copy are also saved under
`data/essays/exports/` with an `AI_REDTEAM_` filename. **Analyze in Detect** sends it to the Detect
page for a full analysis with highlights. The **Saved samples** tab totals every recorded sample:
how many the detector scored, and how often each attack passed as human.

The page uses Gemini's Interactions API with the key in `GEMINI_API_KEY` (the project `.env`), and
tries a configurable fallback list after quota or temporary service errors. Its defaults include
`gemini-3.5-flash-lite`, `gemini-3.6-flash`, `gemini-3.1-flash-lite`, and `gemini-3-flash-preview`.
The older 2.5 Flash models are omitted because Google restricts them for new users. After those it
falls back to a local Ollama model (`gemma3:4b` by default), and the sidebar says whether that can
run. The public deploys leave the page out with `OPENDETECTOR_REDTEAM=off`.

```bash
.venv/bin/python -m opendetector essay.txt
```

The first run downloads the models: GLYPH 0.7 GB, Desklib 1.7 GB, and the Qwen2.5-0.5B pair
about 2 GB. On a GPU (including Apple MPS) they load in bfloat16, so the app needs about
3.5 GB of RAM with all three engines. The eval harness loads one engine at a time, so it
fits on an 8 GB machine. On a bigger machine, set `binoculars_observer`/`binoculars_performer`
in `opendetector/config.py` to a larger base/instruct pair (the paper uses Falcon-7B), then
refit the combiner.

## Evaluate and calibrate

Every number in the report comes from texts the combiner never saw. Data is split by source
document into development (80%) and test (20%), so a text's attacked copies and its AI mirror
always land on the same side.

Thresholds are cross-fitted. Each development text is scored by a model trained on the other
four folds, and the 1% and 5% cut-offs come from those scores on every development human
text, instead of from a small held-out slice. False-positive rates are reported per source
document, because attacked copies of one text aren't independent errors.

Human writing rewritten by an AI paraphraser counts as **AI**: RAID's "human" texts under its
paraphrase attack, and MAGE's `*_human_para` sources (MAGE labels those machine text too).
Catching AI rewrites of human text is part of the job, and the report's `paraphrase` row shows
how well the detector does at it.

**1. Public benchmarks.**

```bash
.venv/bin/python -m eval.featurize raid --out eval/cache/raid.jsonl
```

```bash
.venv/bin/python -m eval.featurize mage --out eval/cache/mage.jsonl --n 3000
```

The RAID run reads 300 random 1 MB slices of RAID-train (about 300 MB, all 11 attacks and 8
domains) instead of downloading 11.8 GB. GLYPH was trained on RAID's non-adversarial data, so
RAID flatters it. Watch the attacked slices and MAGE.

**2. Student essays,** which is where false positives hurt. Download PERSUADE 2.0 (human essays
from grades 6-12, with English-learner status) from
[github.com/scrosseye/persuade_corpus_2.0](https://github.com/scrosseye/persuade_corpus_2.0).
Then write AI essays on the same assignments. The generator defaults to Gemini through its
OpenAI-compatible endpoint, using `GEMINI_API_KEY` from the project's `.env` file (see below):

```bash
.venv/bin/python -m eval.featurize persuade --path persuade_2.0.csv --n 2000 --out eval/cache/persuade.jsonl
```

```bash
.venv/bin/python -m eval.generate_essays --persuade persuade_2.0.csv --n 300 --style student --out data/essays/gemini_student.jsonl
```

```bash
.venv/bin/python -m eval.featurize local --path data/essays/gemini_student.jsonl --out eval/cache/gemini_student.jsonl
```

The free tier allows very few requests on `gemini-3.8-flash`. When its quota runs out, the
generator prints Google's message and switches to `--fallback-model` (default
`gemini-3.5-flash-lite`). Each essay records the model that actually wrote it, and rerunning
the command resumes where it stopped.

Repeat the generation for other model families and for `--style humanize`, which is the attack
you most need to resist. For other providers, pass `--base-url` and `--api-key-env` (OpenAI,
OpenRouter, or a local LM Studio/Ollama server, which needs no key). Any of your own labelled
data works too: a JSONL with `text` and `label` (`human`/`ai`).

To test the detector against AI text that imitates one writer, add `--style-profile` with a
JSON style profile. Its rules and measurements become the system prompt: sentence rhythm,
vocabulary, banned phrases, and mechanical slips such as missing apostrophes. Keep profiles in
`data/styles/`, which git ignores, because a profile of real writing is personal. The profile's
file name joins the attack label (`plain+student_profile`), so the report gives it its own row:

```bash
.venv/bin/python -m eval.generate_essays --persuade persuade_2.0.csv --n 300 --style-profile data/styles/student_profile.json --out data/essays/gemini_profile.jsonl
```

The supplied writer profile is saved as `data/styles/pasted_style_profile.json`. To generate
mirrors in that style, use:

```bash
.venv/bin/python -m eval.generate_essays --persuade persuade_2.0.csv --n 300 --style-profile data/styles/pasted_style_profile.json --out data/essays/gemini_pasted_style.jsonl
```

**API keys** live in a `.env` file in the project root, which git ignores. The scripts load it
automatically, and a variable already exported in your shell takes priority:

```
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
```

**3. Fit and read the report.**

```bash
.venv/bin/python -m eval.fit_combiner eval/cache/*.jsonl
```

This writes `models/combiner.json`, which the app picks up automatically, and
`eval/reports/latest.md`. The report includes:

- AUROC and TPR at 1% and 5% FPR for each engine alone and for the combination
- the real test FPR at the deployed threshold, with a 95% Wilson interval
- FPR and TPR broken down by dataset, domain, attack, generator and group, including the
  English-learner slice

**A low overall FPR with a high English-learner FPR is a failing result.**

**4. Optional: the public RAID leaderboard.**

```bash
.venv/bin/python -m eval.raid_submit --out leaderboard/predictions.json
```

RAID-test has about 672k texts, so run this on a GPU machine. Then open a pull request to
[liamdugan/raid](https://github.com/liamdugan/raid). Declare it if the combiner saw RAID-train.

## Deploy free on Streamlit Community Cloud (slim build)

Community Cloud's free tier caps an app at about 2.7 GB of RAM, too little for the full
ensemble. The **slim** profile runs Desklib alone, the strongest single engine, in float32 on
the CPU. It peaks around 1.5 GB, its scores match the full build's Desklib scores (correlation
1.000), and it has its own combiner, `models/combiner_slim.json`, so its verdicts are still
calibrated. The same `app.py` runs both builds; `deploy/streamlit_cloud/streamlit_app.py` just
pins the slim profile and leaves out the Red team page. The theme comes from
`.streamlit/config.toml` at the repository root. (bfloat16 and int8 were measured and rejected:
bfloat16 is emulated on most CPUs and took minutes per essay, and int8 cut Desklib's AUROC from
0.998 to 0.934.)

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub, and allow access to
   private repositories if asked.
2. Click **Create app**, then **Deploy a public app from GitHub**, and fill in:
   - Repository: `luka751/Open-Detector`
   - Branch: `feat/phase1-ensemble` (or `main` once merged)
   - Main file path: `deploy/streamlit_cloud/streamlit_app.py`
   - Advanced settings, Python version: **3.12**
3. Click **Deploy**. Dependencies come from `deploy/streamlit_cloud/requirements.txt`, which
   installs CPU-only PyTorch. The first analysis downloads Desklib (1.7 GB) and takes a few
   minutes; later ones are much faster.

To run the slim build locally:

```bash
OPENDETECTOR_PROFILE=slim .venv/bin/streamlit run app.py
```

## Deploy to Hugging Face Spaces (needs PRO)

Hugging Face now requires a PRO subscription to run Docker Spaces, even on free CPU hardware.
With PRO, the full ensemble runs on a CPU Space: 16 GB of RAM covers all three engines in full
precision. Sign in once with a **write** token from huggingface.co/settings/tokens, then deploy:

```bash
.venv/bin/hf auth login
```

```bash
.venv/bin/python deploy/huggingface/deploy.py
```

This creates a private Space, `<your-username>/open-detector`, and uploads only `app.py` with its
`ui` package and theme (without the Red team page), the `opendetector` package, the fitted
combiner and the files in `deploy/huggingface/`. No `.env` or API keys are uploaded. Run the same
command again to redeploy after changes. Add `--public` for a
public Space, or change its visibility later in the Space's settings. The first analysis after a
restart downloads about 4 GB of models and takes a few minutes.

## Red-team the detector

`align_engine.py` is the attack this detector has to survive. It has Claude write a text,
scores it with the ensemble, and has Claude rewrite the flagged passages with varied
sentence rhythm and phrasing, until the AI likelihood falls under a target (default 5%) or
it runs out of attempts. It needs Anthropic credentials (`ANTHROPIC_API_KEY`, or `ant auth login`).

```bash
.venv/bin/python align_engine.py "Write a 600-word essay on loneliness in Of Mice and Men" --log data/attacks/mice.jsonl
```

```bash
.venv/bin/python -m eval.featurize local --path data/attacks/mice.jsonl --out eval/cache/attacks.jsonl
```

Every draft goes into the log labelled as AI. The share of prompts that reach the target is
your detector's evasion rate under adaptive attack. The drafts that got through are the
hard positives to train on in Phase 2. The harness loads all three engines, so don't run it
alongside a featurize job on an 8 GB machine.

## Tests

```bash
.venv/bin/python -m pytest -q
```

The tests use stand-in engines, so they need no model downloads.

## Known limits

- English only.
- Sentence highlights reuse the document combiner on ~100-word windows. They show where to
  look; only the document score is calibrated. Paragraph breaks are respected, but inside a
  single paragraph that switches author mid-way, sentences near the switch still blend. Fixing
  that needs a boundary-aware model (Phase 3).
- AI paraphrasing of human writing is the weakest spot (see the `paraphrase` row in the report).
- In testing on an 8 GB Mac, Streamlit crashed once (segfault) under heavy memory pressure
  with all three engines loaded. It didn't recur in later runs. If it happens, switch off
  Binoculars in the sidebar.
- The 0.5B Binoculars pair is weaker than the paper's 7B pair.
- Commercial humanizers remain the hardest case. The way to fix that is training a model on
  humanized text, which is Phase 2 of the roadmap.
- A score is statistical evidence, not proof. Don't use it as the only basis for an academic
  integrity decision.
