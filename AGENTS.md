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
- **Mac app:** `deploy/macos/build_app.py` installs `Open Detector.app`, the same Streamlit app in a
  native window that runs on this checkout. With `--standalone --dmg` it builds the self-contained
  download, `dist/Open-Detector.dmg`, offered on the repo's GitHub releases for anyone to install.

`README.md` is the GitHub front page: what the app does and how to run it. The repo started as a
clone of followsci's "Open Detector" BERT detector. `README_EN.md`,
`README_ZH.md`, `example_usage.py` and `images/` are that original project's and describe a
model this code no longer uses.

## Repo map

| Path | Role |
|---|---|
| `opendetector/` | The detector package. Pure Python except `engines/`, which imports torch. |
| `opendetector/config.py` | `DetectorConfig`: model ids, device/dtype, window sizes, `min_words`, combiner path, and `for_profile("full" \| "slim")`. Also `DATA_ROOT`, the folder that holds what the app writes (`data/` and `.env`): the project folder, or `OPENDETECTOR_HOME` when set. |
| `opendetector/normalize.py` | `normalize(text) -> (clean_text, NormalizationReport)`: strips invisible characters, maps Cyrillic/Greek look-alikes inside Latin text, applies NFKC, collapses whitespace. |
| `opendetector/segment.py` | `split_sentences` (character spans and paragraph index), `sentence_windows` (sentence-aligned sliding windows, optionally kept inside paragraphs), `paragraph_blocks`, `looks_like_prose`, `count_words`. |
| `opendetector/documents.py` | Markdown, Word and PowerPoint text: `extract_document` (sections with their locations), `extract_answer_slots` (question/answer pairs), `rewritable_blocks` (the prose a whole-document rewrite changes: no headings, titles, questions, tables or answer-box pairs) and `replace_document_section`. |
| `opendetector/engines/classifier.py` | `ClassifierEngine.window_logits(texts) -> AI log-odds`; `load_glyph`, `load_desklib`; `DesklibAIDetectionModel` (architecture from its model card). |
| `opendetector/engines/binoculars.py` | `BinocularsEngine.span_scores(text, spans)`. Token statistics come from overlapping 512-token chunks, with the output head run in 32-token blocks to bound memory. Lower score means more AI-like. |
| `opendetector/engines/__init__.py` | `build_engine(name, config)`, `build_engines`, `resolve_device`, `resolve_dtype`, `free_memory`, `ENGINE_NAMES = ("glyph", "desklib", "bino")`. |
| `opendetector/detector.py` | `Detector`: `features(text)` gives document features, `score(text)` a probability, and `analyze(text)` a full `Report` with verdict, probability, engine readings, sentence scores and normalization counts. |
| `opendetector/combiner.py` | `Combiner` (JSON-serialized logistic regression with medians, scaling and thresholds), `contributions` (each feature's term in the log-odds, for the Details chart), `fit_logistic`, `set_thresholds`, `fit_combiner`. Prediction is numpy only. |
| `opendetector/metrics.py` | `auroc`, `tpr_at_fpr`, `conformal_threshold`, `wilson_interval`. |
| `opendetector/env.py` | `load_env()`: loads the gitignored `.env` in `DATA_ROOT` into the environment; variables already set take priority. `save_env()` sets or removes one key there (file mode 600), as the Guide page does. |
| `opendetector/__main__.py` | Command line: `python -m opendetector essay.txt` prints a JSON report. Reads `.md`, `.docx` and `.pptx` like the app (whole document plus every 150-word section); `--answers-only` scores just a worksheet's answer cells; `--profile slim`. |
| `.claude/skills/ai-detect/` | The `/ai-detect` Claude Code skill: scores a file or pasted text through the command line and summarizes the JSON with `summarize.py`. |
| `app.py` | Streamlit entrypoint: page config, logo, theme, the sidebar's detector build, and top navigation between the Detect and Red team pages. `OPENDETECTOR_PROFILE` pins a profile; `OPENDETECTOR_REDTEAM=off` leaves out the Red team page; `OPENDETECTOR_PUBLIC=on` runs it in public mode. |
| `ui/guide.py` | The Guide page, written for people with no technical background: how to check a text and read the result, and a box that checks a Gemini key with Google and saves it with `save_env`. `app.py` opens it first the first time the app runs on a computer (`welcomed` in `ui/prefs.py`); public mode leaves it out. |
| `ui/prefs.py` | What the app remembers on a computer, in `DATA_ROOT/data/ui_prefs.json`: the look and whether the Guide has been shown. `tests/conftest.py` points it at a temporary file for every test. |
| `ui/models.py` | The loaded engines, shared by both pages, every session and batch worker threads. Holds one profile at a time (`get_detector`, `release`); defaults to slim on machines with 8 GB or less. |
| `ui/detect.py` | The Detect page: pasted text or an uploaded `.md`/`.docx`/`.pptx` (with an answers-only toggle for worksheets), verdict card, engine meters, sentence highlights (every sentence or flagged only) with a map of the text, a chart of sentence scores in reading order, a section bar chart, and calibration details with each engine's push on the score. Shows only the engines the profile provides. |
| `ui/redteam.py` | The Red team page: labelled AI samples from a prompt or an uploaded `.md`/`.docx`/`.pptx`, in a style profile picked from `data/styles/*.json` (default `alexandria_profile`), including batch regeneration of every answer box or text block (Markdown, Word or PowerPoint), several blocks per request, Priorities (points over getting past the detector, sounding like the style profile, speed and quota, shown as a diamond, that pick the model, thinking level and batch settings from `models/generation_tuning.json`, or set by hand), with detector scores per block and for the whole document, an *Analyze in Detect* hand-off, and a Saved samples tab summarizing `data/essays/local_redteam.jsonl` by attack. |
| `ui/components.py` | Shared UI: the verdict card, engine meters and highlight HTML with their CSS (the highlights carry a little JavaScript for the tooltip and map clicks), the Altair charts, `keep()` for widget state that survives page switches, and `persistent_upload()`. `TONES` holds the palettes: the `human`, `review` and `ai` fills were checked for colour-blind separation, and in dark mode review and AI highlights also differ by underline style, so keep both if you change them. |
| `ui/style.py` | The two looks: **Glass** (frosted panels over an aurora, ring dial), the default, and **Instrument** (grid, corner-bracket panels, monospace read-outs, needle gauge), picked under Settings at the foot of the sidebar and remembered in gitignored `data/ui_prefs.json` on local runs (`OPENDETECTOR_LOOK` sets the default). Each has a light and a dark mode; Streamlit's own Settings picks the mode. It dresses Streamlit's widgets through `data-testid` hooks and `st-key-od-panel*` containers, so check those after a Streamlit upgrade. `st.html` strips inline SVG, so the verdict dial is drawn as a data-URI image with the look's colours written in. |
| `.streamlit/config.toml`, `ui/assets/` | The base theme (Geist, violet accent, light and dark palettes) and the logos and marks for each look (`logo[-instrument][-dark].svg`, `mark[-instrument].svg`). |
| `models/combiner.json`, `models/combiner_slim.json` | Fitted combiners for the full and slim profiles. The app loads them automatically. |
| `models/generation_tuning.json` | The Red team page's measured generation settings, written by `eval/tune_generation.py`. Without it the page's Priorities fall back to setting the thinking level, blocks per request and requests at once by hand. |
| `eval/datasets.py` | Loaders returning one column schema (`COLUMNS`): RAID, MAGE, PERSUADE and local JSONL/CSV. |
| `eval/featurize.py` | Runs the engines over a dataset and caches document features to `eval/cache/*.jsonl` (gitignored). |
| `eval/fit_combiner.py` | Splits data, fits the combiner, sets thresholds and writes the evaluation report. |
| `eval/reports/latest.md`, `eval/reports/slim.md` | The current evaluation reports for the full and slim combiners. |
| `eval/tune_generation.py` | Measures red-team generation settings (model, thinking level, blocks per request) on a document, four measures each: the detector's AI likelihood of the rewrite, its style match (`StyleYardstick`: how the rewrite ranks among the writer's own passages of `data/styles/<name>/corpus.jsonl` by distance from the profile's measurements), seconds per block, and requests and Gemini's tokens per block, plus each model's free requests a day (read from Gemini's quota errors into `eval/cache/tuning_limits.json`, or `--daily-limit MODEL=N`). By default measures each model per thinking level at the largest group and the group sizes on one model, and predicts the rest by adding effects. Resumable (`eval/cache/tuning.jsonl`); logs every rewrite as labelled AI data to `data/essays/tuning_redteam.jsonl`; writes `models/generation_tuning.json` and `eval/reports/tuning.md`. |
| `eval/generation_settings.py` | The Red team page's Priorities: 12 points over the four measures pick the setting with the highest points-weighted sum of the measures scaled worst-to-best over the table (`choose`); speed points also set requests at once. Quota also counts the share of the model's daily allowance (`daily_requests`) the document at hand would take; an unknown limit costs nothing. `estimate` gives a document's time, requests, tokens and daily limit. Pure Python, shipped in the Mac app. |
| `eval/raid_submit.py` | Writes `predictions.json` for the public RAID leaderboard (about 672k texts; GPU job). |
| `eval/generate_essays.py` | Red-team data: AI "mirror" essays on PERSUADE assignments, via Gemini's Interactions API (default) or any OpenAI-compatible API. When the hosted models fail it switches to a local Ollama model (`--local-fallback`, default `gemma3:4b`). `clean_essay` strips chat preambles and Markdown before saving. `--style-profile` sends a JSON style profile (kept in gitignored `data/styles/`) as the system prompt and adds its name to the attack label. Also holds the helpers the Red team page uses: `generate_with_fallback`, `unload_local_model`, `local_model_problem`, and `thinking_level`, which maps the page's thinking setting to a level each Gemini model accepts (a model that turns a level or the JSON format down is asked again without it). |
| `eval/redteam_batch.py` | The Red team page's batches. `Checkpoint` saves each block to `data/essays/batches/` as it is generated and scored; `BatchWorker` runs a batch on its own thread, so reruns, reconnects and closed tabs don't end it. Blocks go out in groups of neighbouring blocks, one request per group and a few groups at the same time; `parse_group_reply` reads a group's JSON reply, and the page writes any block it leaves out on its own. A stopped or failed batch is exported as it stands and resumes from the first missing block. |
| `align_engine.py` | Red-team harness: Claude writes, the detector scores, and Claude rewrites flagged passages. Every draft is logged as labelled AI data (`--log`). |
| `extract_baseline.py` | Stylometric profile of a writing sample (sentence-length burstiness, TTR/MATTR, punctuation, transitions) written to `style_profile.json`. |
| `update_style_profile.py` | The living style profile `data/styles/alexandria_profile.json`, fed weekly from the writer's Obsidian vault (The Alexandria Archive, in iCloud). Keeps every paragraph ever seen in `data/styles/alexandria/corpus.jsonl` with its ISO week, skips callouts, review notes and pasted lines, and writes recency-weighted measurements (12-week half-life, floor 0.25) blended with the starting profile `data/styles/pasted_style_profile.json`. Snapshots the profile each week to `history/`; `--check` verifies it keeps the template's exact JSON format. |
| `.claude/skills/style-profile/` | The `/style-profile` skill: runs the script, then revises the profile's prose in place so it stacks (keep, add with a week tag, revise only on 3+ weeks of evidence, never delete). The scheduled task `weekly-style-profile` runs it Sundays at 20:00. |
| `deploy/streamlit_cloud/` | `streamlit_app.py` pins the slim profile, turns on public mode and runs `app.py`; `requirements.txt` installs CPU-only PyTorch. |
| `deploy/huggingface/` | Docker Space for the full profile with `deploy.py`. Blocked: Hugging Face now needs PRO for Docker Spaces. |
| `deploy/macos/` | The Mac app. `launcher.py` runs the app's server on 127.0.0.1 (port 8517 when free) and shows it in a pywebview (WebKit) window, with a loading page and an error page (both in the look last picked in the app) that can restart the server, a remembered window size, and a question before quitting during a Red team batch; the server stops itself if the launcher dies. `build_app.py` writes `~/Applications/Open Detector.app`: Info.plist, an icon drawn from `ui/assets/mark.svg`, and a small C executable, compiled with clang, that runs `launcher.py` on the `.venv`'s Python inside the app's own process. `--standalone` instead copies uv's standalone CPython into the bundle, installs `deploy/macos/requirements.txt` at the `.venv`'s versions, drops torch's headers and every package's test suite (`prune_packages`; macOS checks each file of a downloaded app on first launch), copies the code in `APP_FILES`/`APP_PACKAGES`, byte-compiles it, and signs every binary ad hoc; its executable sets `OPENDETECTOR_HOME` to `~/Library/Application Support/Open Detector` and `OPENDETECTOR_PACKAGED=on`, and runs the server too. `--dmg` packs it into `dist/Open-Detector.dmg`. |
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
  - **Public mode** (`OPENDETECTOR_PUBLIC=on`, the live app, and the default whenever the code runs
    from Streamlit Community Cloud's `/mount/src`) serves strangers. The Red team page
    takes the visitor's own key from the sidebar and never reads environment keys, fixes the API
    address to Gemini (a visitor-chosen URL would let them make the server fetch any address),
    and keeps each browser session's samples, profiles and batches in `data/visitors/<id>/`
    through `_store()` and `_own_batches()`. New code that writes or lists saved work must go
    through those, never the module-level paths.
  - **The downloadable app** (`OPENDETECTOR_PACKAGED=on`) keeps its user's key and samples like a
    local run, and offers no local text model by default. It does offer robustness mode, the owner's
    choice; public mode does not, so keep drafting aids out of public mode.
- **What the app writes goes under `DATA_ROOT`,** never `ROOT`: the downloadable app's own bundle is
  signed and must not change, so `PYTHONDONTWRITEBYTECODE` is on there and its code is byte-compiled
  at build time. A module the app newly imports from `eval/` must be added to `APP_FILES`
  (`tests/test_macos_app.py` checks).
- **Git setup:**
  - `origin` is `lukaberadze1/Open-Detector` (public). Pushes use a per-repo credential helper
    that runs `gh auth token --user lukaberadze1`. The global `gh` account is a different one,
    so don't change it.
  - `luka751` is an old private copy. `upstream` is the original project, with no push access.
  - GitHub's front page shows `main`, kept equal to `feat/phase1-ensemble` so visitors see this
    app. After pushing the feature branch, update it with `git push origin feat/phase1-ensemble:main`.
- **Pushing deploys.** Pushing to `feat/phase1-ensemble` should redeploy the live Streamlit app. Its
  entrypoint is `deploy/streamlit_cloud/streamlit_app.py`; the logs show it runs Python 3.14.
  Pushes don't always reach it: on 2026-10-28 Streamlit got every push hook but kept running a
  build cloned an hour earlier. If a change isn't live within a few minutes, sign in, open
  **Manage app** on the live page, check the last "Starting up" time in the logs, and use
  ⋮ → **Reboot app**, which clones the branch afresh.
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
- **The Mac app runs the checkout, not a copy** (the `--standalone` download aside). `Open
  Detector.app` holds absolute paths to this folder's `.venv` and `deploy/macos/launcher.py`, so code
  changes show up the next time it opens, and moving the project means running `build_app.py` again. Its executable must stay a compiled
  program that embeds Python: a shell script that `exec`s python loses the app's process (macOS
  reports pid -1), which breaks accessibility and bringing the open window forward. The launcher
  tells a Red team batch is running from the `caffeinate` child `ui/redteam.py` starts, so keep that
  child if you change how batches keep the Mac awake.
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
- Build or rebuild the Mac app in `~/Applications` (needs the Xcode command line tools):
  ```bash
  .venv/bin/python deploy/macos/build_app.py
  ```
- Build the download, `dist/Open-Detector.dmg` (also needs uv; takes a few minutes), then attach it
  to a GitHub release under that exact name, which README.md's download link points to:
  ```bash
  .venv/bin/python deploy/macos/build_app.py --standalone --dmg
  ```
- Measure the Red team page's generation settings on a document (spends Gemini quota; resumes where it
  stopped), which rewrites `models/generation_tuning.json` and `eval/reports/tuning.md`:
  ```bash
  .venv/bin/python -m eval.tune_generation report.docx
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
