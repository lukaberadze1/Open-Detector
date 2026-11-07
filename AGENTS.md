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
- **A logistic-regression combiner** turns the engine features into one AI likelihood. *Likely AI*
  is set for a 1% false-positive rate; *Likely human* is a fixed 15% or under (`HUMAN_CUTOFF`), with *Needs review* between.
- **It abstains** on texts under 150 words and on non-prose (poetry, lists).
- **It strips evasion tricks** before scoring: zero-width characters and look-alike letters.
- **Two runtime profiles:** `full`, all engines on the best available hardware; and `slim`,
  Desklib alone in float32 on CPU, about 1.45 GB peak, for small hosts.
- **Live app:** https://open-detector.streamlit.app runs the slim profile and the Red team page in
  public mode, deployed from branch `feat/phase1-ensemble` of
  https://github.com/lukaberadze1/Open-Detector (public).
- **Windows app:** `deploy/windows/build_app.py` builds the same app as `dist/Open-Detector-Setup.exe`. It
  runs on GitHub's Windows machines (`.github/workflows/windows-app.yml`), and a published release gets the file.
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
| `opendetector/normalize.py` | `normalize(text) -> (clean_text, NormalizationReport)`: strips invisible characters (zero-width, bidi controls, fillers, variation selectors, tag characters; the joiners inside an emoji are stripped but not counted as a trick), maps Cyrillic/Greek look-alikes inside Latin text, applies NFKC, collapses whitespace. |
| `opendetector/segment.py` | `split_sentences` (character spans and paragraph index), `sentence_windows` (sentence-aligned sliding windows, optionally kept inside paragraphs), `paragraph_blocks`, `looks_like_prose` (line length, then where sentences end, so wrapped text and short sentences on their own lines aren't taken for verse), `join_wrapped_lines` (rejoins hard-wrapped text before scoring; `Detector._prepare` calls it, other text is untouched), `count_words`, `burstiness`, and `group_parts` and `sentence_runs`, which group short sections, answer boxes and paragraphs into parts of 150 words or more so each gets a rating instead of 'too short'. |
| `opendetector/documents.py` | Plain text (blocks split by blank lines, no markup), Markdown, Word and PowerPoint text, leaving out Markdown's front matter and fenced code and reading merged table cells once: `extract_document` (sections with their locations), `extract_answer_slots` (question/answer pairs), `rewritable_blocks` (the prose a whole-document rewrite changes: no headings, titles, questions, tables or answer-box pairs) and `replace_document_section`. |
| `opendetector/forensics.py` | `inspect_document(filename, data)`: what a `.docx` or `.pptx` says about itself, on the Detect page's **File check** tab and in the command line's JSON. Properties and editing history, hidden, white and microscopic text and invisible characters (`flag`), font changes inside paragraphs, little editing time, one editing session (`note`, since these have ordinary explanations). Standard library only, and never part of the score. Uploads are untrusted, so it refuses oversized parts and XML entity declarations. |
| `opendetector/engines/classifier.py` | `ClassifierEngine.window_logits(texts) -> AI log-odds`; `load_glyph`, `load_desklib`, `load_roberta` (off unless `DetectorConfig.roberta_model` is set; `eval.featurize --engines roberta` sets it to `ROBERTA_MODEL`); `DesklibAIDetectionModel` (architecture from its model card). |
| `opendetector/engines/binoculars.py` | `BinocularsEngine.span_scores(text, spans)`, and `scan`, which adds the whole text's raw perplexity (`perp_doc`) and Fast-DetectGPT reading (`fast_doc`, higher is more AI-like) from the same pass. Token statistics come from overlapping 512-token chunks, with the output head run in 32-token blocks to bound memory. Lower score means more AI-like. |
| `opendetector/engines/__init__.py` | `build_engine(name, config)`, `build_engines`, `resolve_device`, `resolve_dtype`, `free_memory`, `ENGINE_NAMES = ("glyph", "desklib", "bino")`. |
| `opendetector/detector.py` | `Detector`: `features(text)` gives document features, `score(text)` a probability, and `analyze(text)` a full `Report` with verdict, probability, engine readings, sentence scores and normalization counts. |
| `opendetector/combiner.py` | `Combiner` (JSON-serialized logistic regression with medians, scaling and thresholds), `contributions` (each feature's term in the log-odds, for the Details chart), `fit_logistic`, `set_thresholds`, `fit_combiner`. Prediction is numpy only. |
| `opendetector/metrics.py` | `auroc`, `tpr_at_fpr`, `conformal_threshold`, `wilson_interval`. |
| `opendetector/zippy.py`, `zippy_seed.txt`, `ZIPPY_LICENSE` | Compression readings ported from Thinkst's ZipPy (MIT): `compression_readings` gives `zippy_lzma` and `zippy_zlib`, higher is more AI-like, on every build with no model. Brotli and ZipPy's multiprocessing are left out. Readings only until a refit includes them: re-featurized caches gain these columns, so pass `eval/fit_combiner.py --features` to keep them out of a fit. |
| `opendetector/env.py` | `load_env()`: loads the gitignored `.env` in `DATA_ROOT` into the environment; variables already set take priority. `save_env()` sets or removes one key there (file mode 600), as the Guide page does. `env_keys(name)` gives every key for a provider: `name` first, then the comma-separated extras in `name + "S"` (`GEMINI_API_KEYS`), which `save_extra_keys` writes. |
| `opendetector/__main__.py` | Command line: `python -m opendetector essay.txt` prints a JSON report. Reads `.md`, `.docx` and `.pptx` like the app (whole document plus its sections, short ones grouped with their neighbours to 150 words or more); `--answers-only` scores just a worksheet's answer cells; `--profile slim`. |
| `.claude/skills/ai-detect/` | The `/ai-detect` Claude Code skill: scores a file or pasted text through the command line and summarizes the JSON with `summarize.py`. |
| `app.py` | Streamlit entrypoint: page config, logo, theme, and the sidebar: navigation between the Detect, Red team and Guide pages, with Settings (the look) at its foot. The detector build is picked in the Detect page's Engines panel and kept across pages like the `keep:` keys. `OPENDETECTOR_PROFILE` pins a profile; `OPENDETECTOR_REDTEAM=off` leaves out the Red team page; `OPENDETECTOR_PUBLIC=on` runs it in public mode. |
| `ui/guide.py` | The Guide page, written for people with no technical background: how to check a text and read the result, and a box that checks a Gemini key with Google and saves it with `save_env`. `app.py` opens it first the first time the app runs on a computer (`welcomed` in `ui/prefs.py`). In public mode it neither opens first nor saves anything: `_public_key_panel` explains where a visitor's key goes instead. |
| `ui/device_storage.py`, `ui/assets/device_storage/` | `device_value`: a value kept in the visitor's own browser (localStorage, this site only), for public mode's **Remember it on this device** key option. The component is plain HTML and JavaScript answering only the app on its own origin, so another site embedding it can't ask for the key. |
| `ui/prefs.py` | What the app remembers on a computer, in `DATA_ROOT/data/ui_prefs.json`: the look and whether the Guide has been shown. `tests/conftest.py` points it at a temporary file for every test. |
| `ui/nav.py` | The app's pages as functions (`detect_page`, `redteam_page`, `guide_page`), so a page can link to another without importing it. The Red team page's visible title is "Adversarial Lab". |
| `ui/models.py` | The loaded engines, shared by both pages, every session and batch worker threads. Holds one profile at a time (`get_detector`, `release`); defaults to slim on machines with 8 GB or less. |
| `ui/detect.py` | The Detect page, in two columns: on the left Source (pasted text or an upload), Engines (build, engine switches, sentence highlights) and Analyze; on the right the result. Pasted text or an uploaded `.md`/`.docx`/`.pptx` (with an answers-only toggle for worksheets), verdict card, engine meters, sentence highlights (every sentence or flagged only) with a map of the text, a chart of sentence scores in reading order, a section bar chart (sections, or a pasted text's paragraphs, rated in parts of 150 words or more through `score_parts`, so short ones are rated with their neighbours; the Red team's Full breakdown regroups its blocks the same way), and calibration details with each engine's push on the score. Shows only the engines the profile provides. Outside public mode the highlights are pickable (`pickable_highlights`): click sentences, shift-click a run, then **Revise selected**, **Revise all Likely AI** or **Revise all flagged** rewrites them in place with the Lab's style profile and models (`_revise_bar`, `redteam.revise_settings` and `revise_chosen`), puts the result in Source and analyzes it again, with Undo. Saved as AI data only with **The source is AI text** on. |
| `ui/redteam.py` | The Red team page. Its settings column runs Mode (**Generate** new text from instructions, or **Regenerate** a document's own text: the whole document, one section or the answer boxes, which map to the `sample`, `blocks`, `section` and `answers` tasks), Input, Priorities, Writing, Detector, then Model & connection folded away, with one button at the foot; the result sits on the right. It makes labelled AI samples from a prompt, pasted text (handled as a `.txt` upload) or an uploaded `.txt`/`.md`/`.docx`/`.pptx`, in a style profile: the user's own from `data/styles/*.json` first where there is one, otherwise (and always in public mode) the built-in human one (`_styles`; an upload can't take a built-in's name), including batch regeneration of every answer box or text block (plain text, Markdown, Word or PowerPoint), several blocks per request, Priorities (points over getting past the detector, sounding like the style profile, speed and quota, shown as a diamond, that pick the model, thinking level, temperature and batch settings from `models/generation_tuning.json`, or set by hand), an "own writing" toggle that also saves the source's originals as human-labelled pairs (label 0, same `group_key`, once each, through `_human_rows`), each sample's settings saved with it (`_settings`), a "Show the model the writer's own passages" toggle (offered when the profile's folder has writing in the chosen register, off by default; `writer_samples`, never in public mode), "Rewrite the draft in the writer's sentences" for samples from a prompt (`Plan.restyle`, on by default when the profile has sentence shapes), "Revise what the detector flags", on by default for samples and document rewrites (`Plan.revise`, `REVISE_BY_DEFAULT`, `REVISE_ROUNDS` rounds at most of `revise_flagged` or `revise_anchored`, passage by passage until it scores under `WIN_BELOW` or stalls (`REVISE_PATIENCE`), with `REVISE_KEEPING`, `meaning_kept`, `REVISE_CANDIDATES` rewrites a passage (the lowest-scoring kept, `stock_phrases` breaking ties) and `_style_fits`, so a revision that loses the passage's names and numbers, invents new ones, flips a negation, or with a user's own profile moves away from it, is dropped; never in public mode, scores saved as `revise_rounds`), "How closely to follow the original" for document rewrites (`Plan.structure`: light edit, same structure, writer's sentences (the default) with `Plan.shapes` from `_writer_shapes` (never in public mode), or free rewrite; each block's structure match and words kept are saved with it through `_followed`), with detector scores per block and for the whole document, a **Full breakdown** under the score that draws the Detect page's result in place (`_breakdown`, `detect.breakdown` and `detect.show_result`; it scores the sentences the first time it is opened), an *Analyze in Detect* hand-off, and a Saved samples tab summarizing `data/essays/local_redteam.jsonl` by attack. |
| `ui/components.py` | Shared UI: the verdict card, engine meters and highlight HTML with their CSS (the highlights carry a little JavaScript for the tooltip and map clicks; `pickable_highlights` draws them as an `st.components.v2` component, unisolated so the app's CSS applies, that reports picked sentences back), the Altair charts, `keep()` for widget state that survives page switches, and `persistent_upload()`. `TONES` holds the palettes: the `human`, `review` and `ai` fills were checked for colour-blind separation, and in dark mode review and AI highlights also differ by underline style, so keep both if you change them. |
| `ui/style.py` | The two looks: **Glass** (frosted panels over an aurora, ring dial), the default, and **Instrument** (grid, corner-bracket panels, monospace read-outs, needle gauge), picked under Settings at the foot of the sidebar and remembered in gitignored `data/ui_prefs.json` on local runs (`OPENDETECTOR_LOOK` sets the default). Each has a light and a dark mode; Streamlit's own Settings picks the mode. It dresses Streamlit's widgets through `data-testid` hooks and `st-key-od-panel*` containers, so check those after a Streamlit upgrade. `st.html` strips inline SVG, so the verdict dial is drawn as a data-URI image with the look's colours written in. |
| `.streamlit/config.toml`, `ui/assets/` | The base theme (Geist, violet accent, light and dark palettes) and the logos and marks for each look (`logo[-instrument][-dark].svg`, `mark[-instrument].svg`). |
| `models/combiner.json`, `models/combiner_slim.json` | Fitted combiners for the full and slim profiles. The app loads them automatically. |
| `models/generation_tuning.json` | The Red team page's measured generation settings, written by `eval/tune_generation.py`. Without it the page's Priorities fall back to setting the thinking level, blocks per request and requests at once by hand. |
| `models/recipe.json`, `eval/recipe.py` | The Priorities panel's recipe: which measured features a document rewrite uses (`masked`, `roundtrip-<language>`), picked by the same points as the model settings. `build` compares each setup in the overnight runs with its own model's baseline, runs on human-written sources (`+human-sources`) apart, and those on AI text decide, since a human original is nothing to get through; `choose` scales getting under the line, style distance, minutes and requests worst to best, and never picks a setup keeping under `MEANING_FLOOR` (70%) of the original's names and numbers while one keeping more is measured: meaning is a floor, not a priority. The panel sits near the top of the page; **Write my own settings** (`OWN_SETTINGS`) brings back the hand settings, holding names and numbers and More attacks, which are hidden otherwise. Never in public mode, and only for whole-document rewrites with the models, which is what was measured. `python -m eval.recipe` rebuilds it, as the overnight runner's closing `recipe` stage does each night (`eval/reports/recipe.md`). In `APP_FILES`. |
| `eval/datasets.py` | Loaders returning one column schema (`COLUMNS`): RAID, MAGE, PERSUADE and local JSONL/CSV. `pairs_to_local` turns AI and human pairs (ClearRewriter's, in gitignored `data/pairs/`) into `load_local`'s rows. `load_local` leaves out an AI-labelled red-team row that is a copy of its original (`redteam_batch.copied_row`): the writer's own text, not AI text. |
| `eval/featurize.py` | Runs the engines over a dataset and caches document features to `eval/cache/*.jsonl` (gitignored). |
| `eval/fit_combiner.py` | Splits data, fits the combiner, sets thresholds and writes the evaluation report. |
| `eval/reports/latest.md`, `eval/reports/slim.md` | The current evaluation reports for the full and slim combiners. |
| `eval/reports/extra_signals.md` | Why burstiness, Fast-DetectGPT, raw perplexity and RoBERTa aren't in the score, measured over 20 splits and the red-team samples. |
| `eval/tune_generation.py` | Measures red-team generation settings (model, thinking level, temperature, blocks per request) on a document's prose blocks, four measures each: the detector's AI likelihood of the rewrite, its style match (`StyleYardstick`: how the rewrite ranks among the writer's own passages of `data/styles/<name>/corpus.jsonl` by distance from the profile's measurements; a share with under `MIN_CASES` cases, like one 'an' against the writer's 98% 'a', isn't measured, and `text_distance` gives the distance itself), seconds per block, and requests and Gemini's tokens per block, plus each model's free requests a day (read from Gemini's quota errors into `data/model_limits.json`, shared with the page, or `--daily-limit MODEL=N`). By default measures each model per thinking level at the largest group and base temperature, and the group sizes and temperatures on one model, and predicts the rest by adding effects; a model that ignores temperature (`takes_temperature`) gets none. Trials from an earlier sample stay in the cache but only the latest sample's are fitted; a model that times out twice is skipped. `StyleYardstick.build(..., register)` measures the finalized register against the finished pieces when there are enough, the notes otherwise. `style_examples` picks a few of the writer's most typical passages in a register (notes from the corpus, finished work only from `finalized/`, so the two never mix) and `StyleYardstick.checklist` turns the measurements into counts (sentences and commas per 100 words, short and long sentences); `--examples N` sends them with the profile, as the page's switch does. `--structure` (default `writer`, as the page) sets how closely rewrites follow the original, and the report adds their structure match and words kept; runs other than free add `+<structure>` to the profile label, so they are cached apart. Resumable (`eval/cache/tuning.jsonl`); logs every rewrite as labelled AI data to `data/essays/tuning_redteam.jsonl`; writes `models/generation_tuning.json` and `eval/reports/tuning.md`. `--provider groq` or `openrouter` measures that provider's models and writes its own cache, table (`eval/cache/generation_tuning_<provider>.json`) and report, so the table the page reads stays as it was. |
| `eval/tune_prompts.py` | Measures standard generation, new samples from a prompt alone, as `tune_generation.py` measures document rewrites: a set of prompts (`PROMPTS`, finished or informal) written per model and repeat in a profile's register, each sample scored for AI likelihood, verdict and style match, and logged as labelled AI data. `--restyle` (the page's default) rewrites each draft in the writer's sentences; `--framing`, `--plan` and `--old-voice` are the variants that made no difference; `--revise N` adds up to N rounds of `revise_flagged` and records the score before and after; `--plain-words` adds `redteam_batch.plainer_words`, which asks the model to replace the words the writer never uses (`writer_vocabulary`) and stays out of the page: on 2026-11-02 it cut those words by 40-100% but replaced them with wrong ones ('printer' became 'bad machine'); `--examples N` sends the writer's own passages. Resumable (`--out`). |
| `eval/tune_revise.py` | Measures revising a whole-document rewrite against the detector: each repeat writes one rewrite and revises it (`--methods`: by sentence scores or by hottest document window, through `revise_blocks`, or `anchored`, through the page's `revise_anchored` with each written block against its original, labelled `+anchored`; `--tag` adds to the label, and `--mage N --human-sources` writes human-written documents the detector reads as human, the page's own case, to `data/essays/sources/human/`), with optional instructions on top of the profile (`--note`, `NOTES`), and `--as-page` for the page's revision (`REVISE_KEEPING`, `keeps_facts`, `style_guard`; `--no-guard` leaves the guard out). `--rounds 0` measures the first draft alone. Several documents give the pattern one text can't (`eval/reports/revise.md`, `--full` for the full detector); `--mage N` writes AI documents from MAGE. Waits out an overloaded model, resumes from `--out`, logs every text as labelled AI data. |
| `eval/tune_drafts.py` | Compares first drafts: each change on its own (`SETUPS`: temperature, top-p, top-k, round trip and `NOTES`) through `tune_revise --rounds 0`, best first in `eval/reports/drafts.md`. Drafts starting under 80% reached the 10% bar far more often. Keep `data/essays/sources/fresh/` out of this screening: those documents are the final proof. |
| `eval/human_swaps.py` | Word swaps run on every reply of the built-in human profile only (`styled_clean(..., swaps=True)`, never in public mode): stock phrases, AI-favoured words, some synonyms and contractions from `styles/human_default/swaps.json`, ported from ClearRewriter. On 10 saved rewrites they took the mean from 84% to 72% (2026-11-02). Pure Python, in `APP_FILES`. |
| `eval/generation_settings.py` | The Red team page's Priorities: 12 points over the four measures pick the setting with the highest points-weighted sum of the measures scaled worst-to-best over the table (`choose`); speed points also set requests at once. Quota also counts the share of the model's daily allowance (`daily_requests`, plus `learned_limits` from `data/model_limits.json`, which the page's batches add to through `record_limits` except in public mode) the document at hand would take; an unknown limit costs nothing. A setting also carries its temperature. `estimate` gives a document's time, requests, tokens and daily limit. Pure Python, shipped in the Mac app. |
| `eval/evasion_mode.py` | The Red team page's adaptive challenge mode (`generate_until_target`): up to three labelled AI drafts ranked on detector score and style match, every draft saved as AI data. Never in public mode. |
| `eval/text_attacks.py` | Attacks on finished AI text, run after writing and before revising: `round_trip` through another language (`PIVOTS`; two requests for about every 400 words, and a paragraph that loses a number or changes length stays as it was) and `reorder_clauses` (`turned_around` moves a subordinate clause to the other side, no words change, no requests; it leaves alone a sentence where the clause is the object or names a time: "I wonder if...", "the year when..."). `free_round_trip` goes through Google Translate's public web address with no model or quota, offered only with the built-in profile: its bot check refused the first request on 2026-11-02, so it is unmeasured. `run_attacks` applies them in turn and `attack_label` adds them to the sample's attack. Offered under the page's **More attacks** with top-p and top-k, off by default and never in public mode, and in `tune_prompts.py` and `tune_revise.py` as `--attack`, `--top-p` and `--top-k`. None of it is measured yet. In `APP_FILES`. |
| `eval/local_translate.py`, `eval/bench_hop.py` | The local translation hop, an idea from lynote-ai/humanize-text (MIT): MarianMT (`Helsinki-NLP/opus-mt-*`, about 570 MB on disk each direction) writes the final English, so no LLM chooses it, with no key or quota. `text_attacks.local_round_trip` (`localtrip-<language>`) goes there and back sentence by sentence, and a sentence that loses a number or name, changes length or keeps foreign letters stays as it was (`_sound`); `rewrite_trip` (`rewritetrip-<language>`) has a model rewrite in the other language first. On the page under **More attacks**, never in public mode or the download, and in `tune_revise.py` and `tune_prompts.py` as `--attack`. `bench_hop` measures it on the math benchmark's texts, no API. First look on 4 texts (2026-11-05, slim, German): 0.979 to 0.926, 0.990 to 0.164, 0.999 to 0.995, 0.996 to 0.971, words kept 81%, with visible meaning damage ('THE DEADLY MANTIS' became 'THE DEATH MANTIS'; 'between it' became 'between him'), and `_sound` can't see that. Four texts settle nothing. |
| `eval/score_demo.py` | Grades the slim detector on humanize-text's demo data, kept in gitignored `data/pairs/humanize_text_demo/`: 50 AI texts, their human rewrites, and six rewrites each by that project's own model. Read-only, about 35 minutes, a stage of `eval/overnight.py`. Not training data: its license is unclear. |
| `eval/math_attack.py` | The no-model command-line attack on existing AI text, for when quota runs out: the deterministic rules, `human_swaps`, clause reordering and sentence splits and merges (`vary_lengths`), saved as labelled AI to `data/essays/math_redteam.jsonl`. `guided_attack` tries each edit on each passage and keeps it only if the detector then scores the whole text lower; `fitted_swaps` allows a synonym only if its word pairs occur in human writing (`styles/human_default/bigrams.json.gz`, counts from MAGE's test split, `eval/build_bigrams.py`), which stopped swaps like 'create sure'. It cleans its input first (`sanitize`: invisible characters, look-alike letters, NFC, layout kept) and records ZipPy's readings before and after (`compression_change`). On the page as the Mode panel's **Engine**, never in public mode: API (models), Math (no model at all, edits supplied text: `_math_sample`, or `generate_group` for a batch), Combo (models, then the script); and as an API batch's last resort when every model including Ollama has failed (model `MATH_MODEL`). `_math_edit` allows swaps only with the built-in profile. A math edit is saved as AI only when the user turns on **The source is AI text** (`Plan.source_is_ai`), whatever share of the words it keeps: rules can take a short text under `COPIED`, and it is still its author's. On machines with 8 GB or less a batch leaves Ollama out when its prompts need more than `SMALL_MACHINE_LOCAL_CONTEXT`, which a big style profile does even for a short document. The phrase rules in `opendetector/core/rules/ngrams.json` marked `mined` come from `eval/mine_ngrams.py` (`eval/reports/ngrams.md`), with replacements written by hand; a phrase must be common in human text too, since MAGE repeats one source across generators. Synonyms come from the hand-reviewed `SAFE_SYNONYMS` (the ported table's hold in some senses only: 'old' to 'earlier'), stock phrases skip `UNSAFE_PHRASES`, and `_keeps_length` drops a pass that changes a text's length too much. `--score --power` adds `eval/gradient_attack.py`. Weak: with these safe swaps, 12 MAGE AI texts (2026-11-04, slim) go from a mean of 0.997 to about 0.99 with none under 0.5. More rule types won't help: on those texts, reversing the sentence order, 10% typos, all lowercase and dropping half the commas each left the median at 0.997 or above with none under 0.5. |
| `eval/bench_math.py` | The math script's standard test: slim detector, no API, a fixed set of MAGE AI texts, printing scores, model runs and time. `CachedScorer` (`eval/math_attack.py`) keeps window logits in memory and here on disk, so a rerun only pays for text it hasn't seen. A text of 350 words or fewer is one window, so an edit always costs a model run on its first sight. `--band LOW HIGH` takes pool texts starting in that range and reports log-odds. The script works on borderline text, not saturated text (2026-11-04, slim): 17 of the 200 pool texts start in 0.6-0.95 and go from a mean of 0.869 to 0.672, 6 of the 12 Likely AI ones leaving that verdict, against a mean of 0.997 to 0.993 and none on 12 saturated texts. `eval/blamed_words.py` ranks words by what masking them does, over each text's first window. `eval.math_attack --score` skips texts under 150 words, since the detector abstains there and nothing can guide the edits; `--by-group` joins saved blocks into documents by `group_key`, `--baseline` only prints words and scores, `--band` limits the search to a score range and `--budget` caps its detector passes. Active and passive voice conversion was tried with spaCy on simple sentences and dropped (2026-11-04): only 2 sentences in 29 texts qualified, and one came out as 'Tears of joy were cried by my daughter'. |
| `eval/math_transfer.py` | Edits the benchmark's borderline texts guided by slim, then scores them before and after on the full build. On 17 texts (2026-11-04) the full build's mean goes from 0.877 to 0.705 and 6 of 12 Likely AI texts leave that verdict, about the same as slim, but the full combiner's weight is mostly on Desklib, so this is no independent check. About an hour on the 8 GB Mac. |
| `eval/overnight.py` | The overnight tests in order of what they decide, one model-heavy job at a time. Each stage has a time limit and its own log, a crash is tried once more, and a failure is noted and passed by along with what needs its result. `python -m eval.overnight` resumes (finished stages are skipped, `--again STAGE` reruns one); `--until 10:00` stops starting work before the closing stages; `--smoke` runs each stage tiny in its own folder. Gemini comparisons (`gemini_setups`, all on AI-written documents: the page's anchored loop with names and numbers held, the same without, and the first on `MORE_DOCUMENTS`; multi-start, the German round trip and the old loop were measured on 6-7 Nov and are out) run in rounds of one repeat per setup, so a spent quota leaves them equally measured, then the next of `MODELS` (Gemini, then OpenRouter; Groq's free tier refuses requests this size) starts its comparison afresh, and a model that refuses the request (`UNUSABLE_EXIT`) is out for the night. A spent model is tried again `RECHECK` (an hour) after it ran out, or at the daily reset if sooner, and once every other stage is done `catch_up` comes back to the rounds quota, overload or time cut short, waiting for a model to be worth trying while there is time, before the closing stages. `--day` runs the short stages (`DAY`) most useful first; a run started `--at` a time holds no lock while it waits, then waits for a day run to finish. Stage limits are fitted to the default `--until 07:40`. Reads and writes `eval/cache/overnight/`: `BRIEF.md`, rewritten after every stage, is the thing to read. |
| `eval/trace.py` | A record of every step of a text (`tune_revise --trace`, which the overnight runner passes): each version with scores by window and sentence, each rewrite tried and why it was kept or dropped (`meaning_problem`), each request with route, tokens, thought summary and full prompt (`generate_essays.OBSERVER`). Holds whole texts and the style profile, so it stays in gitignored folders and out of the page and public mode. |
| `eval/overnight_summary.py` | The overnight scoreboard (`summary.md`): every setup by model and document. A pass is at or under `HUMAN_CUTOFF` with `FACTS_KEPT`% of names and numbers kept; comparisons use only runs both setups reached, with a sign test across documents; an independent detector scores each draft and final text; plus whether a lower first draft ends lower within a document, and why rewrites get dropped (`outcomes` from `revise_blocks`). |
| `eval/detector_anatomy.py` | What the slim detector reads: each of `CHANGES` (lowercase, shuffles, masked word classes, shorter text) on saturated and borderline AI texts and human ones, what really masking a word does, and the raw Desklib score the combiner needs (`required_raw_score`). Written to `anatomy.md`. No API. |
| `eval/trial_compare.py` | The shipped slim combiner against a trial one on the same texts: the test side of each public cache and the saved red-team samples (`collect_samples`, ids by text) split into guided and not. Writes nothing to `models/`. |
| `eval/gradient_attack.py` | Experimental white-box word search, not on the page: `gradient_search` swaps the words in the hottest document window that masking shows matter most (`masked_pushes`; gradient times embedding came out near zero and tracked real masking no better than chance on 4 texts, 2026-11-05), using reviewed synonyms that fit both neighbours (`fitting_words`) and sit close in its embeddings; `power_attack` runs it around `guided_attack`. With `SAFE_SYNONYMS` it changes almost nothing, since the rule stage uses them up. Candidates from word pairs alone reached 0.730 with 3 under 0.5 by changing meaning ('sang the song' to 'sang the season'), which shows how few words can move Desklib. Untested on the full build. |
| `eval/contextual_attack.py` | The math script's model-backed stage (`--contextual` on `eval.math_attack --score` and `eval.bench_math`), in the downloads only as unused code, since its models and vectors stay out. `propose` asks a fill-in-the-blank model for words, `MeaningGuard` checks each result against the original sentence (similarity and entailment), `ranked_places` measures what removing words sampled evenly from the hottest windows does, and `beam_search` keeps the lowest log-odds versions. `Quality` adds part of speech and inflection (spaCy, lemminflect), grammar (CoLA) and, for rewrites, no new names or unfounded words; `close_word` needs counter-fitted vectors and refuses listed antonyms, both in gitignored `data/vendor/counter-fitting/`. `paraphrase_stage` (`--paraphraser t5`, `pegasus` or `dipper`) rewrites sentences under the same checks; DIPPER needs a GPU and has not been run; Pegasus reads 60 tokens at most, so longer sentences stay as they are. `propose` adds the counter-fitting project's PPDB synonyms to the model's words. `python -m eval.contextual_attack` tests `MeaningGuard` on PAWS: at its strictest useful bar it still passed 60% of false paraphrases (2026-11-04). On the page as **Use language models too** under Math and Combo, never in public mode or the download. `eval.bench_math --independent` also scores with a detector the search never sees (`OUTSIDE`: Tabularis's ModernBERT by default, pinned, since `roberta` missed 65% of the reference set's AI text); `--out` saves each text as it finishes and a rerun with the same settings resumes from it; `--quick` is a smaller search for screening only (on 2 texts it found none of the full search's gain, 2026-11-05); `--no-cache` times a run fairly. `CachedScorer.many` scores versions in one batch and `PATIENCE` stops a stalled search. First runs (2026-11-04, slim): word swaps barely move saturated text; T5 rewrites took 2 saturated texts from 0.985 to 0.311 before the stricter checks, with visible meaning drift, so that is a lead and not a result. |
| `eval/train_adapter.py` | Trains a small open model (`DEFAULT_BASE`, Qwen2.5-1.5B-Instruct) on the writer's notes. `prepare` joins the vault corpus's paragraphs into `passages` and has a hosted model rewrite each in polished prose twice (`PARAPHRASE`, resumable, into gitignored `data/adapters/pairs.jsonl`), with each note's passages on one side of the held-out split (`held_out`). `train --method lora` learns the polished version back into the writer's passage; `--method dpo` starts from a LoRA (`--start`) and prefers the writer's passage to the model's second rewrite, with the starting model's log-probabilities worked out once (`with_references`) instead of keeping a frozen copy. `write` rewrites a text with an adapter, logging each block as labelled AI data to `data/essays/adapter_redteam.jsonl` and never a copy of its original; `export` merges an adapter and writes an Ollama Modelfile. Only the notes go in, never `finalized/`. Needs an NVIDIA GPU unless `--force`. Imports torch and peft only inside the functions that train. Kept out of `ui/`, `app.py` and `APP_FILES` (a test checks), since an adapter holds the writer's writing. Run end to end only on a tiny random model so far (2026-11-02), so it has never trained on real data. |
| `eval/fact_mask.py` | `FactMask`: names, numbers, citations, URLs, math and code go out as `[F1]`-style placeholders with a legend, and come back after the rewrite, so a model can't drop or change them (the prompt alone kept 55%, `eval/reports/revise.md`). `rewrite_group` uses it with `plan.mask_facts`: a reply that loses a placeholder is asked for again, then written on its own unmasked, where `meaning_kept` still checks it. On the page as **Hold names and numbers exactly**, on by default since the 6-7 Nov overnight run (`MASK_BY_DEFAULT`: 78% of names and numbers kept against 38%, as low a score, 2 rounds against 3) and never in public mode; `eval.tune_revise --mask-facts` measures it under its own `+masked` label. In `APP_FILES`. |
| `eval/humanizer_attack.py` | Public humanizers run over MAGE AI texts no combiner was fitted on, every output labelled AI in `data/essays/humanizer_redteam.jsonl` under its source's `group_key` (sources in `.sources.jsonl` beside it). `academic-control` and `stealth-rules` need no model; `stealth-bart` and `cive-bart` are about 1.6 GB each and load safetensors only; `stealth-gemma` uses local Ollama. The outside detector `--picker` picks best-of-N winners, so none is guided by this detector, and `eval/reports/humanizers.md` gives each one's before and after. Rows carry `words_retained`, not `words_kept`: the source is AI, so a light edit stays AI data (`copied_row`). Not on the page or in the downloads. Ported code's licenses are in `eval/HUMANIZERS_LICENSE`. |
| `eval/redteam_report.py` | Scores every saved red-team sample of 150+ words of prose (`local_redteam.jsonl`, `tuning_redteam.jsonl`, `adapter_redteam.jsonl`, `math_redteam.jsonl`, `humanizer_redteam.jsonl`), copies of their originals left out, with the shipped combiner and writes `eval/reports/redteam.md`: the share of AI samples the detector misses and of human originals it flags, with Wilson intervals, by attack, model, thinking, temperature and blocks per request. Read-only; caches scores in `eval/cache/redteam_scores.jsonl` by text, profile and combiner. |
| `eval/reference_benchmark.py` | The reference standard: slim, full and RoBERTa (and outside detectors through `--export` and `--external`) graded against text of known origin, never each other, written to `eval/reports/reference.md`. MAGE's test side of the combiner's split (its group keys are per text, so one source can sit on both sides), DAIGT V2 if it's in `data/pairs/`, HC3 if it's in the Hugging Face cache, the saved samples (`guided` when the detector was used in writing them, hence the transfer gap), and fresh text in `data/reference/{human,ai}/<genre>/*.txt`. `--export` never writes saved samples or `data/reference/`. Resumes from `eval/cache/reference_scores.jsonl`. |
| `eval/raid_submit.py` | Writes `predictions.json` for the public RAID leaderboard (about 672k texts; GPU job). |
| `eval/check_providers.py` | Tries Groq and OpenRouter with the keys in `.env`: is the key good, what models are on offer (`--list`), and does each model write plain text and JSON with a style profile as big as a real request's. Saves nothing. |
| `eval/generate_essays.py` | Red-team data: AI "mirror" essays on PERSUADE assignments, via Gemini's Interactions API (default), Groq or OpenRouter (`PROVIDERS`, `--provider`), or any OpenAI-compatible API. `chat` gives those two their own handling: reasoning kept out of the reply (`_provider_options`, and `<think>` stripped from what comes back), the JSON shape asked for and dropped if a model turns it down (`_drop_rejected`), a reply cut off at its token limit treated as an error, OpenRouter's errors inside a 200, its 402 and free daily cap as `QuotaExhausted`, and in `generate_with_fallback` only a 401 stops the run, since one model's 400 or 403 says nothing about the next. OpenRouter skips hosts that keep or train on prompts unless `OPENROUTER_DATA_COLLECTION=allow`. `KeyPool` holds several keys from different accounts: `generate_with_fallback` takes one in place of a key and runs each model on every key it has left, in turn, before the next model (`_on_every_key`); a key out of quota is out for that model only, a rejected key for all, `RATE_STRIKES` per-minute refusals in a row count as spent, and while every key is only rate-limited the request waits (`pause`, at most `MAX_HOLDS` times) instead of moving down. One key behaves as before. A 429 that says "per day" is a spent quota whatever wait it asks for, and a short 429 raises `RateLimited`. When the hosted models fail it switches to a local Ollama model (`--local-fallback`, default `gemma3:4b`). `clean_essay` strips chat preambles and Markdown before saving. `--style-profile` sends a JSON style profile (kept in gitignored `data/styles/`) as the system prompt and adds its name to the attack label. Also holds the helpers the Red team page uses: `generate_with_fallback`, `unload_local_model`, `local_model_problem`, `style_instructions` (a registers-v2 profile sends only the chosen register, `--register`, plus `shared`, less any habit or signature word seen once or tagged for the other register; older profiles render whole; the writer's passages and counts, `--examples`, come last), `styled_clean`, `temper_quirks` and `temper_misspellings` (models overdo any slip they are told about, so spaces before punctuation, double spaces, spaced brackets and lowercase 'i' are cut back to the writer's rate in `measurements.json`, and recurring misspellings to the share of a word's uses the writer misspells, as each reply is cleaned; they are only ever removed), and `thinking_level`, which maps the page's thinking setting to a level each Gemini model accepts (a model that turns a level or the JSON format down is asked again without it). |
| `eval/key_usage.py` | Each Gemini key's requests per model per quota day (midnight Pacific), counted by `generate_essays.chat` on every reply and set to none left on a quota error, whose stated limit it also records (`record_limits`). Gemini's API doesn't say what a key has left, so this is this computer's own count: requests made elsewhere with the key aren't seen. Keys are stored as a hash in `DATA_ROOT/data/key_usage.json`; public mode records nothing. `left` gives what the keys panel draws as bars. Standard library only, in `APP_FILES`. |
| `eval/gemini_fallback.py` | `generate_with_fallback`'s step between the hosted models and Ollama: the Gemini website through the unofficial `gemini_webapi` client (`chat_web`, model `WEB_MODEL`), signed in with the session in the owner's Firefox, read fresh for each request (`firefox_session`, browser-cookie3). `web_available` needs `GEMINI_WEB_FALLBACK=on` in `.env` and keeps it off in public mode and the downloadable app. Temporary chats, the nearest model (`web_model`), `thinking == "high"` as extended thinking, and the system prompt as a gem (`_gem_for`, one per distinct prompt in the account, in the message if that fails). One signed-in client for the run on its own event loop, `WEB_PARALLEL` requests at once. Three tries (`RETRY_WAITS`) before Ollama, counting a non-JSON reply, one past `max_tokens` or the website's error message as failures. The website takes no temperature, top-p or top-k: a temperature of `HIGH_TEMPERATURE` or more only picks among the replies offered (`pick_reply`). The client sends no such field, so don't guess at its request slots. Errors give the exception's type only, so cookies can't reach a message. |
| `eval/redteam_batch.py` | The Red team page's batches. `Checkpoint` saves each block to `data/essays/batches/` as it is generated and scored; `BatchWorker` runs a batch on its own thread, so reruns, reconnects and closed tabs don't end it. Only prose blocks are rewritten (`worth_rewriting`: labels, names, dates and numbers stay). Blocks go out in groups of neighbouring blocks filled by words (`group_by_words`, 30-word places), one request per group and a few groups at the same time; `parse_group_reply` reads a group's JSON reply, and the page writes any block it leaves out on its own. A stopped or failed batch is exported as it stands and resumes from the first missing block. Rewrites can follow the original whoever wrote it (`STRUCTURES`), or, in `writer`, a skeleton borrowed from the profile's writer (`borrowed_skeleton` over `paragraph_shapes` of their paragraphs, which `tune_generation.writer_shapes` reads for a register: a run of their sentences about the block's length, without lines under `MIN_PLANNED` words, listed as a sentence plan of lengths and punctuation, never their words; `off_plan` has a rewrite short of its plan or with a cut-off sentence asked for again once, `again_note` choosing the note), since an AI original's even sentences survive any rewrite in place: `original_text` shows the model the block's sentences numbered with their lengths, `follow_original` puts the original's paragraph breaks and spacing back around the new sentences when the counts match, and `rewrite_measures` gives the structure match (sentence lengths, punctuation and paragraphs, in order) and the share of the original's words kept. A rewrite keeping `COPIED` (90%) or more of the original's words is the original handed back (`copied`; with a style profile a free rewrite of the writer's own text nearly always is): the page and the benchmark ask for it again once with `AGAIN`, the page never saves a copy as AI, and `copied_row` keeps saved copies out of the evaluation. Standard samples come from `sample_prompt` (shared with the page), and with a profile `restyle` rewrites the draft passage by passage (`draft_jobs`) in the writer's sentences, through the same `rewrite_group`, with plans `RESTYLE_STRETCH` longer and drafts `DRAFT_STRETCH` longer since models write short into a borrowed plan: written from a prompt alone a sample keeps the model's own content plan, which the detector reads whatever voice it is in. `person_instruction` says nothing without source text (it once kept "I" and "you" out of every sample written from a prompt). `_written_tries` asks for all of a passage's versions in one request, and `restyle(candidates, score)` uses that for the first draft (`FIRST_DRAFT_TRIES`), so revising should run less often; neither is measured yet. `revise_flagged(above, max_requests)` rewrites only passages at a chosen score and stops at a request limit (`REVISE_REQUESTS`). The Lab's **Revise a text above a percentage** panel (`_revise_panel`, reached from a Detect result, never in public mode) runs it on supplied text, with a model or, through `contextual_attack.revise_above`, on this computer. `revise_flagged` closes the loop on a finished sample: it rewrites only the passages the detector's sentence scores flag (`passage_spans`, `passage_scores`), in fresh borrowed plans with `REVISE`, keeping each rewritten passage only if the whole text then scores lower (one detector pass per passage), so one bad passage no longer costs a round its good ones. `revise_anchored` is what the page now runs on a whole-document rewrite (`drifted` also drops a try losing a name or number the writer's original and the passage share: the per-try 80% of `keeps_facts` compounded, and runs that took 8 or 9 rounds kept 12-19% of the facts on 6-7 Nov; `ANCHORED_FIRST_ROUND` tries in its first round, all in the same requests, and plans stretched by `REVISE_STRETCH` since revisions came back short; tries start from the original only when the detector reads it as human, under `ANCHOR_BELOW`, since an AI original's wording is what gets caught): each round takes every passage that reads as AI in the `CONTEXT_WORDS` around it (`chunk_scorer`, a chunk read as one document window), not a capped few, gives each `ANCHORED_CANDIDATES` tries, half starting from the writer's original of its block (`ANCHORED`, the job's `anchor`) and half as before, scores each try on its own surroundings, prefers the try keeping most of the original's words among those already under half the bar, drops one that leaves its block a copy (`COPIED`), allows a pronoun the original itself uses, undoes a round the whole document reads as worse, and keeps going when one request fails (an overloaded model). On yoo.docx's saved 2,262-word first draft at 89% (2026-11-06, slim, gemini-3.1-flash-lite) 5 runs of 5 reached the 10% bar, in 1 or 2 rounds (ending at 5-9.5%), about 23 requests and 3 minutes a round on the 8 GB Mac with nothing else running, where revise_blocks had taken 8 rounds and about an hour to reach 16%; the result kept 65-71% of the original's words, against the draft's 73%, so it holds closeness rather than adding to it. One document, so not yet a rate. `revise_selected` is the Detect page's revise: only the reader's picked passages (`selected_spans`, runs of picked sentences inside a paragraph, cut near 80 words), each try scored in its `CONTEXT_WORDS` surroundings as `revise_anchored` does, a round the whole text reads as worse undone, stopping when every picked passage reads as human or a round keeps nothing. `revise_blocks` does the same for a whole-document rewrite; given `score_many` (the page passes `CachedScorer(detector).many`) it scores each passage's tries in one batch and runs the model only on document windows it hasn't seen, the same figures (on a 2,262-word document, about 7 minutes a round became about 1.5 on the 8 GB Mac, 2026-11-05), and given `saved` it hands over each round as it ends: the page writes it to the checkpoint (`revise_partial`), so a quit or crash keeps the rounds done and a resume goes on from them. |
| `align_engine.py` | Red-team harness: Claude writes, the detector scores, and Claude rewrites flagged passages. Every draft is logged as labelled AI data (`--log`). |
| `extract_baseline.py` | Stylometric profile of a writing sample (sentence-length burstiness, TTR/MATTR, punctuation, transitions) written to `style_profile.json`. |
| `update_style_profile.py` | The living style profile `data/styles/alexandria_profile.json`, fed weekly from the writer's Obsidian vault (The Alexandria Archive, in iCloud). Keeps every paragraph ever seen in `data/styles/alexandria/corpus.jsonl` with its ISO week, skips callouts, review notes and pasted lines, and writes recency-weighted measurements (12-week half-life, floor 0.25). `register_numbers` keeps the two registers apart: notes from the vault, finalized from the starting profile (`BASELINE`) blended with `finished_counts`: the finished pieces in `data/styles/alexandria/finalized/` (`finished_sources`: `.md`, `.txt`, `.docx`, `.pptx` or `.pdf`, weights from `weights.json`, blank templates in `templates/` subtracted, only the writer's own prose kept by `own_prose`) and the vault notes the writer rated Silver or Gold (`note_ladders`, weights in `LADDER`). Snapshots the profile each week to `history/`; `--check` verifies it keeps `style_profile_template.json`'s exact format, habit entries included. |
| `styles/human_default_profile.json`, `styles/human_default/` | The built-in human style profile, the Red team page's default (`BUILTIN_STYLE_DIR`, `DEFAULT_STYLE`, shown as "Human (built in)"), and listed before the user's own. No one writer's: measured from 1,750 human texts by different writers in MAGE (finished work: news, science abstracts, Wikipedia, stories; quick notes: argument posts, explanations, reviews) by `eval/build_human_profile.py`, which writes numbers only: `measurements.json` (each register whole and its spread across writers, for the prose), `shapes.json` (sentence shapes for "Writer's sentences", `writer_shapes`) and `yardstick.json` (the style match's target, spread and passage distances by passage size, `StyleYardstick.measured`). Holding no one's writing, it works on a public deploy too, shapes included. Writing as one measured writer rather than the average was tried and made samples more AI-like (eval/reports/style_profile_ab.md), so it describes the average. In `APP_FILES`. |
| `eval/build_human_profile.py` | Rebuilds `styles/human_default/` from MAGE's human texts in the Hugging Face cache (250 texts of 150+ words per source, texts whose punctuation was lost left out). The profile's prose is written from `measurements.json` like any other profile's. |
| `style_profile_template.json` | The style profile format ("registers-v2"): two registers, `finalized` (finished work) and `notes` (quick notes), each with rhythm, voice, structure, punctuation, slips (recurring misspellings with counts, one-off kinds) and `habits` (habit, rate, status established/emerging/once, seen, example), plus `shared` (always, never, signature vocabulary, banned phrases, hedging rules). Each field's text says what goes there. |
| `.claude/skills/style-profile/` | The `/style-profile` skill: runs the script, then revises the profile in place so it stacks (keep, add with a week tag, revise only on 3+ weeks of evidence, never delete). Vault writing goes to the notes register only; each habit gets a rate and a status from its counts (established: 3+ weeks or 10+ times; emerging: 2 weeks or 3-9; once: never a rule), and misspellings seen 3+ times are listed with counts while one-offs are kept as kinds. The scheduled task `weekly-style-profile` runs it Sundays at 20:00. |
| `deploy/streamlit_cloud/` | `streamlit_app.py` pins the slim profile, turns on public mode and runs `app.py`; `requirements.txt` installs CPU-only PyTorch. |
| `deploy/huggingface/` | Docker Space for the full profile with `deploy.py`. Blocked: Hugging Face now needs PRO for Docker Spaces. |
| `deploy/macos/` | The Mac app. `launcher.py` runs the app's server on 127.0.0.1 (port 8517 when free) and shows it in a pywebview (WebKit) window, with a loading page and an error page (both in the look last picked in the app) that can restart the server, a remembered window size, and a question before quitting during a Red team batch; the server stops itself if the launcher dies. `build_app.py` writes `~/Applications/Open Detector.app`: Info.plist, an icon drawn from `ui/assets/mark.svg`, and a small C executable, compiled with clang, that runs `launcher.py` on the `.venv`'s Python inside the app's own process. `--standalone` instead copies uv's standalone CPython into the bundle, installs `deploy/macos/requirements.txt` at the `.venv`'s versions, drops torch's headers and every package's test suite (`prune_packages`; macOS checks each file of a downloaded app on first launch), copies the code in `APP_FILES`/`APP_PACKAGES`, byte-compiles it, and signs every binary ad hoc; its executable sets `OPENDETECTOR_HOME` to `~/Library/Application Support/Open Detector` and `OPENDETECTOR_PACKAGED=on`, and runs the server too. `--dmg` packs it into `dist/Open-Detector.dmg`. |
| `deploy/windows/`, `.github/workflows/windows-app.yml` | The Windows download. `build_app.py` copies uv's standalone CPython into a stage folder, installs `deploy/macos/requirements.txt` at the versions in `constraints.txt` (`--pin` writes them from the `.venv`, and a test fails when they drift), copies the code in the Mac build's `APP_FILES`/`APP_PACKAGES`, byte-compiles it, and compiles with MSVC a small `Open Detector.exe` (icon `app.ico`, drawn from the Mac icon) that loads the app's Python and runs `deploy/macos/launcher.py`, which works on both systems; given arguments it runs them as python would. `installer.iss` packs that with Inno Setup into `dist/Open-Detector-Setup.exe`: per-user, no administrator password, and installs Microsoft's WebView2 where it is missing. The workflow runs the tests on Windows, builds it, and runs `smoke_test.ps1`: install, score a text with the installed app (about the Mac's score), open the window, and check the server stops when the app is killed. Data goes to `%LOCALAPPDATA%\Open Detector`. Publishing a GitHub release runs it and attaches the file; the app is unsigned, so Windows SmartScreen warns once. |
| `deploy/remote_test/` | Runs the tests and the detector-against-generator checks on another computer and packs the results to send back. That kit holds the owner's writing. `build_package.py --human` packs one without it, which `run_human.py` uses to test the built-in profile in three setups (`START_HUMAN.md`). |
| `tests/` | pytest suite with stand-in engines: no model downloads, about 10–25 s. |
| `DETECTOR.md` | User guide: setup, evaluation workflow, deployment, red-teaming, known limits. |
| `ruff.toml`, `pytest.ini`, `requirements.txt` | Lint config, test config, and development dependencies (Python 3.12 venv in `.venv`). |
| `requirements-attack.txt` | Optional packages for the Gemini website fallback and the math script's language-model stage. Out of `requirements.txt` because `gemini_webapi` conflicts with the Windows pins, which stopped the v1.2.0 Windows build. |
| `requirements-train.txt` | `requirements.txt` plus `peft` and `accelerate`, for `eval/train_adapter.py` on a GPU machine. Not in the `.venv`, the app or the downloads. |

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
   `{glyph,desklib}_max`, `bino_doc` (whole text), and `bino_min` (lowest document window). Also `fast_doc` and
   `perp_doc` (full build) and `burst_cv`, `burst_adj`, `zippy_lzma` and `zippy_zlib` (every build). The combiner uses only the features it was
   fitted on, so these are readings until a refit includes them: the Details table says which count.
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
    takes the visitor's own key under Model & connection and never reads environment keys or writes the key
    anywhere (it lasts while the tab is open, or in the visitor's own browser if they ask:
    `ui/device_storage.py`), never sends the writing kept beside a profile (`writer_samples`), fixes the API
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
- **Providers on the page.** Model & connection's Provider sets the URL, models and key variable (`_provider_changed`), and a missing Groq or OpenRouter key can be pasted there (`_key_box`, into `.env`). With Gemini, `_gemini_keys` adds keys from other accounts (checked with Google, kept in `GEMINI_API_KEYS`); with more than one, `Connection.pool` is a `KeyPool` for the run, a batch logs each key spent or rejected, and the priorities count each model's daily limit once per key (`generation_settings.with_keys`). The keys popover shows each key's requests left today per model as bars (`_key_allowance`, from `eval/key_usage.py`), and its label the total left on the first model. The priorities' table holds Gemini's measurements, so with another provider its settings are set by hand. Public mode has none of this: its address stays fixed to Gemini.
- **Gemini quota.** The free tier allows very few requests on `gemini-3.8-flash`, so
  `eval/generate_essays.py` switches to `gemini-3.5-flash-lite` on a spent quota, and both
  generators fall back to `gemma3:4b` on Ollama after that. Ollama must be running (`ollama serve`)
  with the model pulled; the Red team page's Model & connection says when it isn't.
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
  child if you change how batches keep the Mac awake. On Windows the launcher looks for a named mutex
  held by the same keep-awake process instead (`BATCH_MUTEX`).
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
  to a GitHub release under that exact name, which README.md's download link points to (publishing the
  release also builds and attaches `Open-Detector-Setup.exe`, the Windows download, on GitHub's machines):
  ```bash
  .venv/bin/python deploy/macos/build_app.py --standalone --dmg
  ```
- After changing package versions in the `.venv`, update what the Windows build installs:
  ```bash
  .venv/bin/python deploy/windows/build_app.py --pin
  ```
- Measure the Red team page's generation settings on a document (spends Gemini quota; resumes where it
  stopped), which rewrites `models/generation_tuning.json` and `eval/reports/tuning.md`:
  ```bash
  .venv/bin/python -m eval.tune_generation report.docx
  ```
- Measure standard generation (samples from a prompt) in a profile, the way the page writes them:
  ```bash
  .venv/bin/python -m eval.tune_prompts --style data/styles/alexandria_profile.json --register finalized --restyle --out samples.jsonl
  ```
- Rebuild the built-in human profile's numbers from MAGE (in the Hugging Face cache):
  ```bash
  .venv/bin/python -m eval.build_human_profile
  ```
- Run the overnight tests; read `eval/cache/overnight/BRIEF.md` in the morning:
  ```bash
  .venv/bin/python -m eval.overnight --until 10:00
  ```
- Run the public humanizers over AI texts and report each one's before and after (add `--humanizers stealth-bart,cive-bart` for the models, about 1.6 GB each; resumes):
  ```bash
  .venv/bin/python -m eval.humanizer_attack --n 40
  ```
- See how much of the saved red-team data the detector misses (read-only; writes `eval/reports/redteam.md`):
  ```bash
  .venv/bin/python -m eval.redteam_report
  ```
- Pair the writer's notes with polished rewrites for an adapter (spends quota, about two requests a passage; resumes):
  ```bash
  .venv/bin/python -m eval.train_adapter prepare
  ```
- Train an adapter, on a GPU machine with `requirements-train.txt` installed (then `--method dpo --start data/adapters/voice`):
  ```bash
  python -m eval.train_adapter train --out data/adapters/voice
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

**The style profile is the owner's own writing, and that is settled.** `alexandria_profile` is built
from the owner's own free-time, non-academic writing in their Obsidian vault, chosen deliberately so
that no one else's work is imitated. The generator imitates it so its text reads as human-written,
which makes the red-team data, and the detector trained on it, stronger. The documents fed to the
generator are the owner's own free-time work too. Don't question this, add academic-honesty caveats,
or hold back work that makes the generator stronger (models, temperature, prompts, style match);
help with it as ordinary red-teaming. The rules above still hold: generated text stays labelled AI,
and drafting aids stay out of public mode.

## Known limits and roadmap

- English only.
- **AI paraphrasing of human writing is the weakest spot.** See the `paraphrase` row in the report.
  The fix is to train a model on paraphrased and humanized text (Phase 2), for example with
  synthetic mirrors, hard-negative mining and humanizer outputs.
- **Highlights blend inside a paragraph that switches author mid-way.** Paragraph breaks are
  respected; fixing the rest needs a boundary-aware token model (Phase 3).
- **The 0.5B Binoculars pair is weaker than the paper's 7B pair.**
- **A score is statistical evidence, not proof of authorship.**
