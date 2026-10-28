# Guide for AI agents working on Open Detector

Read this before changing anything. It explains what the project does, where everything
lives, how a text is scored, how the detector is evaluated and calibrated, and the
invariants that are easy to break. Human-facing documentation is in `DETECTOR.md`.

## What this project is

An AI-generated-text detector, built as a calibrated ensemble:

- **Three engines** score overlapping windows of the whole text:
  - **GLYPH**: a DeBERTa-v3-base classifier
  - **Desklib**: a DeBERTa-v3-large classifier, the strongest single engine
  - **Binoculars**: a zero-shot detector built on a Qwen2.5-0.5B base/instruct pair
- **A logistic-regression combiner** turns the engine features into one AI likelihood. Its
  thresholds are set for a 1% (*Likely AI*) and 5% (*Needs review*) false-positive rate.
- **It abstains** on texts under 150 words and on non-prose (poetry, lists).
- **It strips evasion tricks** before scoring: zero-width characters and look-alike letters.
- **Two runtime profiles:** `full`, all engines on the best available hardware; and `slim`,
  Desklib alone in float32 on CPU, about 1.45 GB peak, for small hosts.
- **Live app:** https://open-detector.streamlit.app runs the slim profile and the Red team page in
  public mode, deployed from branch `feat/phase1-ensemble` of
  https://github.com/lukaberadze1/Open-Detector (public).

`README.md` is the GitHub front page: what the app does and how to run it. The repo started as a
clone of followsci's "Open Detector" BERT detector. `README_EN.md`,
`README_ZH.md`, `example_usage.py` and `images/` are that original project's and describe a
model this code no longer uses.

## Repo map

| Path | Role |
|---|---|
| `opendetector/` | The detector package. Pure Python except `engines/`, which imports torch. |
| `opendetector/config.py` | `DetectorConfig`: model ids, device/dtype, window sizes, `min_words`, combiner path, and `for_profile("full" \| "slim")`. |
| `opendetector/normalize.py` | `normalize(text) -> (clean_text, NormalizationReport)`: strips invisible characters, maps Cyrillic/Greek look-alikes inside Latin text, applies NFKC, collapses whitespace. |
| `opendetector/segment.py` | `split_sentences` (character spans and paragraph index), `sentence_windows` (sentence-aligned sliding windows, optionally kept inside paragraphs), `paragraph_blocks`, `looks_like_prose`, `count_words`. |
| `opendetector/documents.py` | Markdown, Word and PowerPoint text: `extract_document` (sections with their locations), `extract_answer_slots` (question/answer pairs), `rewritable_blocks` (the prose a whole-document rewrite changes: no headings, titles, questions, tables or answer-box pairs) and `replace_document_section`. |
| `opendetector/engines/classifier.py` | `ClassifierEngine.window_logits(texts) -> AI log-odds`; `load_glyph`, `load_desklib`; `DesklibAIDetectionModel` (architecture from its model card). |
| `opendetector/engines/binoculars.py` | `BinocularsEngine.span_scores(text, spans)`. Token statistics come from overlapping 512-token chunks, with the output head run in 32-token blocks to bound memory. Lower score means more AI-like. |
| `opendetector/engines/__init__.py` | `build_engine(name, config)`, `build_engines`, `resolve_device`, `resolve_dtype`, `free_memory`, `ENGINE_NAMES = ("glyph", "desklib", "bino")`. |
| `opendetector/detector.py` | `Detector`: `features(text)` gives document features, `score(text)` a probability, and `analyze(text)` a full `Report` with verdict, probability, engine readings, sentence scores and normalization counts. |
| `opendetector/combiner.py` | `Combiner` (JSON-serialized logistic regression with medians, scaling and thresholds), `fit_logistic`, `set_thresholds`, `fit_combiner`. Prediction is numpy only. |
| `opendetector/metrics.py` | `auroc`, `tpr_at_fpr`, `conformal_threshold`, `wilson_interval`. |
| `opendetector/env.py` | `load_env()`: loads the gitignored `.env` into the environment. Variables already set take priority. |
| `opendetector/__main__.py` | Command line: `python -m opendetector essay.txt` prints a JSON report. Reads `.md`, `.docx` and `.pptx` like the app (whole document plus every 150-word section); `--answers-only` scores just a worksheet's answer cells; `--profile slim`. |
| `.claude/skills/ai-detect/` | The `/ai-detect` Claude Code skill: scores a file or pasted text through the command line and summarizes the JSON with `summarize.py`. |
| `app.py` | Streamlit entrypoint: page config, logo, theme, the sidebar's detector build, and top navigation between the Detect and Red team pages. `OPENDETECTOR_PROFILE` pins a profile; `OPENDETECTOR_REDTEAM=off` leaves out the Red team page; `OPENDETECTOR_PUBLIC=on` runs it in public mode. |
| `ui/models.py` | The loaded engines, shared by both pages, every session and batch worker threads. Holds one profile at a time (`get_detector`, `release`); defaults to slim on machines with 8 GB or less. |
| `ui/detect.py` | The Detect page: pasted text or an uploaded `.md`/`.docx`/`.pptx` (with an answers-only toggle for worksheets), verdict card, engine readings, sentence highlights, section scores and calibration details. Shows only the engines the profile provides. |
| `ui/redteam.py` | The Red team page: labelled AI samples from a prompt or an uploaded `.md`/`.docx`/`.pptx`, in a style profile picked from `data/styles/*.json` (default `alexandria_profile`), including batch regeneration of every answer box or text block (Markdown, Word or PowerPoint), with detector scores per block and for the whole document, an *Analyze in Detect* hand-off, and a Saved samples tab summarizing `data/essays/local_redteam.jsonl` by attack. |
| `ui/components.py` | Shared UI: the verdict card and highlight HTML with their CSS, `keep()` for widget state that survives page switches, and `persistent_upload()`. |
| `.streamlit/config.toml`, `ui/assets/` | The theme (Inter, indigo accent, light and dark palettes) and the logo. |
| `models/combiner.json`, `models/combiner_slim.json` | Fitted combiners for the full and slim profiles. The app loads them automatically. |
| `eval/datasets.py` | Loaders returning one column schema (`COLUMNS`): RAID, MAGE, PERSUADE and local JSONL/CSV. |
| `eval/featurize.py` | Runs the engines over a dataset and caches document features to `eval/cache/*.jsonl` (gitignored). |
| `eval/fit_combiner.py` | Splits data, fits the combiner, sets thresholds and writes the evaluation report. |
| `eval/reports/latest.md`, `eval/reports/slim.md` | The current evaluation reports for the full and slim combiners. |
| `eval/raid_submit.py` | Writes `predictions.json` for the public RAID leaderboard (about 672k texts; GPU job). |
| `eval/generate_essays.py` | Red-team data: AI "mirror" essays on PERSUADE assignments, via Gemini's Interactions API (default) or any OpenAI-compatible API. When the hosted models fail it switches to a local Ollama model (`--local-fallback`, default `gemma3:4b`). `clean_essay` strips chat preambles and Markdown before saving. `--style-profile` sends a JSON style profile (kept in gitignored `data/styles/`) as the system prompt and adds its name to the attack label. Also holds the helpers the Red team page uses: `generate_with_fallback`, `unload_local_model`, `local_model_problem`. |
| `eval/redteam_batch.py` | The Red team page's batches. `Checkpoint` saves each block to `data/essays/batches/` as it is generated and scored; `BatchWorker` runs a batch on its own thread, so reruns, reconnects and closed tabs don't end it. A stopped or failed batch is exported as it stands and resumes from the next block. |
| `align_engine.py` | Red-team harness: Claude writes, the detector scores, and Claude rewrites flagged passages. Every draft is logged as labelled AI data (`--log`). |
| `extract_baseline.py` | Stylometric profile of a writing sample (sentence-length burstiness, TTR/MATTR, punctuation, transitions) written to `style_profile.json`. |
| `update_style_profile.py` | The living style profile `data/styles/alexandria_profile.json`, fed weekly from the writer's Obsidian vault (The Alexandria Archive, in iCloud). Keeps every paragraph ever seen in `data/styles/alexandria/corpus.jsonl` with its ISO week, skips callouts, review notes and pasted lines, and writes recency-weighted measurements (12-week half-life, floor 0.25) blended with the starting profile `data/styles/pasted_style_profile.json`. Snapshots the profile each week to `history/`; `--check` verifies it keeps the template's exact JSON format. |
| `.claude/skills/style-profile/` | The `/style-profile` skill: runs the script, then revises the profile's prose in place so it stacks (keep, add with a week tag, revise only on 3+ weeks of evidence, never delete). The scheduled task `weekly-style-profile` runs it Sundays at 20:00. |
| `deploy/streamlit_cloud/` | `streamlit_app.py` pins the slim profile, turns on public mode and runs `app.py`; `requirements.txt` installs CPU-only PyTorch. |
| `deploy/huggingface/` | Docker Space for the full profile with `deploy.py`. Blocked: Hugging Face now needs PRO for Docker Spaces. |
| `tests/` | pytest suite with stand-in engines: no model downloads, about 10–25 s. |
| `DETECTOR.md` | User guide: setup, evaluation workflow, deployment, red-teaming, known limits. |
| `ruff.toml`, `pytest.ini`, `requirements.txt` | Lint config, test config, and development dependencies (Python 3.12 venv in `.venv`). |

If a nested `Open-Detector/` directory exists locally, it is a separate untracked clone and
not part of this project.

## How a text gets scored (`Detector.analyze`)

1. **Normalize.** `normalize()` returns the clean text, and every offset in the report refers to
   it. Removed hidden characters and homoglyphs are counted and shown to the user.
2. **Split into sentences.** `split_sentences()` handles abbreviations, initials, quotes and
   newlines. Each `Sentence` records its paragraph, meaning which blank-line-separated block it is in.
3. **Abstain** if the text is under `min_words` (150) words, giving verdict `insufficient_text`,
   or if `looks_like_prose()` is False, giving verdict `not_prose`. No model runs in either case.
4. **Build two kinds of windows:**
   - **Document windows:** about 350 words with a 175-word stride, anchored to the end of the
     text. These produce the calibrated features.
   - **Local windows:** about 100 words with a 50-word stride, kept inside paragraphs of at least
     60 words (shorter paragraphs merge with a neighbour first). These produce the sentence
     highlights only.
5. **Score with the engines.** The classifiers return a log-odds value per window. Binoculars
   scores the whole text once and then any span from its token statistics.
6. **Compute features:** `{glyph,desklib}_mean` (word-weighted over document windows),
   `{glyph,desklib}_max`, `bino_doc` (whole text), and `bino_min` (lowest document window).
7. **Combine and decide.** `Combiner.predict_proba` gives p. The verdict is `likely_ai` if
   p > `thresholds["ai"]`, `needs_review` if p > `thresholds["review"]`, else `likely_human`.
   Without a combiner file, the fallback is the mean of the classifiers' sigmoid scores, with
   placeholder cut-offs 0.9 and 0.6, and the report says `calibrated: False`.
8. **Score sentences.** Each local window is scored as a small document through the same
   combiner, and each sentence gets the mean over the windows covering it. Only the document
   score is calibrated.

## Profiles and hardware

- **`full`** (the default `DetectorConfig()`) runs every engine on `device="auto"` (cuda, then
  mps, then cpu) with `dtype="auto"` (bfloat16 on GPU, float32 on CPU). With all three engines
  it needs about 3.5 GB of RAM.
- **`slim`** (`DetectorConfig.for_profile("slim")`) runs Desklib only, CPU, float32, batch size 2,
  with `models/combiner_slim.json`. Its scores match the full build's Desklib scores
  (correlation 1.000). Alternatives were measured and rejected:
  - bfloat16 on CPU is emulated and took about 10 minutes per essay
  - int8 dynamic quantization cut Desklib's AUROC from 0.998 to 0.934
  - a bfloat16 embedding table used more memory than float32, which transformers maps
    straight from the model file
- **The development machine is an 8 GB Mac.** Eval runs load one engine at a time. Don't run two
  model-heavy jobs at once: the machine thrashes, and Streamlit once segfaulted under that
  pressure.

## Evaluation and calibration

**Data** (`eval/datasets.py`). Every loader returns
`id, text, label (0 = human, 1 = AI), dataset, domain, model, attack, group, group_key`.

- **RAID:** `load_raid` reads random 1 MB byte ranges of the 11.8 GB Hugging Face CSV (300 by
  default) and resyncs on UUID row starts. The CSV is sorted by domain and attack, so never
  stream it from the top. Range requests retry with backoff.
- **MAGE:** `load_mage`. MAGE's own `label` uses 1 = human, so labels are derived from `src` and
  cross-checked.
- **PERSUADE 2.0:** `load_persuade`, from a local CSV that must be downloaded manually.
- **Your own data:** `load_local` reads JSONL or CSV with `text` and `label`.
- **`group_key`** ties together every version of one source document: its attacked copies, its AI
  mirror. Splits are made by `group_key`, so near-duplicates never leak into the test set.
- **AI rewrites of human text count as AI.** This covers RAID's "human" rows under the paraphrase
  attack (relabelled in `eval/fit_combiner.py`) and MAGE's `*_human_para` sources.

**Featurize** (`python -m eval.featurize <raid|mage|persuade|local> --out eval/cache/X.jsonl`):

- Runs one engine at a time and caches progress to `X.jsonl.<engine>.partial`, so it resumes.
- Writes `words`, `prose` and the features for every text.
- Options: `--engines`, `--dtype`, `--n`, `--raid-chunks`, `--min-words`, `--limit`, and
  `--annotate-prose` (adds the `prose` flag to an older cache without re-scoring).

**Fit** (`python -m eval.fit_combiner eval/cache/*.jsonl`):

1. Drops texts under 150 words and non-prose texts, which is what the deployed detector abstains on.
2. Splits by `group_key` hash: development 80%, test 20%.
3. Fits the logistic regression on development. Thresholds come from 5-fold out-of-fold scores of
   the development human texts, using a conformal quantile.
4. Reports AUROC and TPR at 1% and 5% FPR for each engine and for the combination, plus the
   deployed-threshold FPR per source document with Wilson intervals, broken down by
   dataset, domain, attack, model and group.

Options: `--features` (restrict to a subset), `--out`, `--report`.

To regenerate both shipped combiners after the caches change:

```bash
.venv/bin/python -m eval.fit_combiner eval/cache/raid.jsonl eval/cache/mage.jsonl
```

```bash
.venv/bin/python -m eval.fit_combiner eval/cache/raid.jsonl eval/cache/mage.jsonl --features desklib_mean,desklib_max --out models/combiner_slim.json --report eval/reports/slim.md
```

## Invariants and gotchas

- **What requires a refit, and what doesn't.** The combiner is fitted to the distribution of
  document features. Changing the engines, model ids, dtype, the document-window sizes, or
  normalization means re-featurizing and refitting. Local windows, highlights and app UI can
  change freely.
- **Feature order.** `Combiner.features` fixes the order. Missing features are imputed with
  training medians, and the app warns when engines are switched off.
- **Never edit code under a running featurize job.** `eval/featurize.py` imports the engine code
  only after the dataset download, so editing `opendetector/` while a run is starting can
  crash it.
- **Keep the `prose` column out of the features.** `eval/fit_combiner.py` treats every column
  not in `META` as a feature, so new metadata columns must be added to `META`.
- **Secrets.** API keys live in `.env`, gitignored, read through `opendetector/env.py`.
  `.env.example` lists the variables with empty values; keep it free of real keys.
  - Never commit keys or print them.
  - Tests must not read the real `.env`; `tests/test_generate_essays.py` patches `load_env`.
  - The repo is public.
  - **Public mode** (`OPENDETECTOR_PUBLIC=on`, the live app) serves strangers. The Red team page
    takes the visitor's own key from the sidebar and never reads environment keys, fixes the API
    address to Gemini (a visitor-chosen URL would let them make the server fetch any address),
    and keeps each browser session's samples, profiles and batches in `data/visitors/<id>/`
    through `_store()` and `_own_batches()`. New code that writes or lists saved work must go
    through those, never the module-level paths.
- **Git setup:**
  - `origin` is `lukaberadze1/Open-Detector` (public). Pushes use a per-repo credential helper
    that runs `gh auth token --user lukaberadze1`. The global `gh` account is a different one,
    so don't change it.
  - `luka751` is an old private copy. `upstream` is the original project, with no push access.
  - GitHub's front page shows `main`, kept equal to `feat/phase1-ensemble` so visitors see this
    app. After pushing the feature branch, update it with `git push origin feat/phase1-ensemble:main`.
- **Pushing deploys.** Pushing to `feat/phase1-ensemble` redeploys the live Streamlit app. Its
  entrypoint is `deploy/streamlit_cloud/streamlit_app.py`, on Python 3.12.
- **Gemini quota.** The free tier allows very few requests on `gemini-3.8-flash`, so
  `eval/generate_essays.py` switches to `gemini-3.5-flash-lite` on a spent quota, and both
  generators fall back to `gemma3:4b` on Ollama after that. Ollama must be running (`ollama serve`)
  with the model pulled; the Red team page's sidebar says when it isn't.
- **Memory with a local model.** On the 8 GB Mac, `gemma3:4b` (about 3 GB) and the full detector
  don't fit together. The Red team page generates every block before scoring any and unloads the
  local model in between. The app starts on the slim build on machines with 8 GB or less, and both
  pages share one loaded profile.
- **Widget state across pages.** Streamlit drops the state of widgets the current page doesn't
  draw. Widgets keyed `keep:…` survive a page switch because `app.py` writes those keys back on
  every run; give them their starting value with `ui.components.keep()`, never a `value=` argument.
  File uploaders can't be kept that way, so pages use `persistent_upload()`.
- **Unicode in source is deliberate.** Cyrillic and Greek look-alike letters and curly quotes in
  `opendetector/normalize.py`, `opendetector/segment.py` and some tests are intentional;
  `ruff.toml` exempts those files.

## Common tasks

Each of these is one command.

- Set up the environment:
  ```bash
  uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
  ```
- Run the tests:
  ```bash
  .venv/bin/python -m pytest -q
  ```
- Lint:
  ```bash
  .venv/bin/ruff check
  ```
- Run the app (Detect and Red team pages):
  ```bash
  .venv/bin/streamlit run app.py
  ```
- Run it as the public deploy does, slim and in public mode:
  ```bash
  OPENDETECTOR_PROFILE=slim OPENDETECTOR_PUBLIC=on .venv/bin/streamlit run app.py
  ```
- Score one file from the command line (`.txt`, `.md`, `.docx` or `.pptx`; add `--answers-only`
  for a worksheet, or use `/ai-detect <file>` in Claude Code):
  ```bash
  .venv/bin/python -m opendetector essay.txt
  ```

**Adding an engine:**

1. Write a loader that returns an object with `name` and either `window_logits(texts)` or
   `span_scores(text, spans)`.
2. Register it in `build_engine` and `ENGINE_NAMES`, add a model-id field to `DetectorConfig`,
   and extend `Detector.feature_names`, `_doc_features` and `_window_features`.
3. Featurize, refit, and compare the per-engine rows in the report.
4. Add stub-engine tests; don't download models in tests.

## Scope

This project builds and evaluates a detector. The generation tools (`align_engine.py`,
`eval/generate_essays.py`, `ui/redteam.py`) exist to produce adversarial test data, and they
log what they write as labelled AI text. Keep them that way.

## Known limits and roadmap

- English only.
- **AI paraphrasing of human writing is the weakest spot.** See the `paraphrase` row in the report.
  The fix is to train a model on paraphrased and humanized text (Phase 2), for example with
  synthetic mirrors, hard-negative mining and humanizer outputs.
- **Highlights blend inside a paragraph that switches author mid-way.** Paragraph breaks are
  respected; fixing the rest needs a boundary-aware token model (Phase 3).
- **The 0.5B Binoculars pair is weaker than the paper's 7B pair.**
- **A score is statistical evidence, not proof of authorship.**
