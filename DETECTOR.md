# Open Detector: ensemble detector and eval harness

```
text ─► normalize ─► sentences ─► overlapping windows ─┬─► GLYPH (DeBERTa-v3-base)   ─┐
        (zero-width,  (spans)     doc: ~350 words      ├─► Desklib (DeBERTa-v3-large) ├─► combiner ─► calibrated p ─► verdict
         homoglyphs)              local: ~100 words    └─► Binoculars (Qwen pair)     ─┘  (logistic    (thresholds set on
                                                                                         regression)  held-out human text)
```

**Mission.** This project is for scientific and engineering work: measuring AI-text detection
rigorously, calibrating it with stated error rates, and stress-testing it with labelled synthetic
samples so weaknesses can be found and fixed. It is not for deceiving readers or evading
integrity rules, and a score is evidence, not proof of authorship.

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
git clone -b feat/phase1-ensemble https://github.com/lukaberadze1/Open-Detector.git && cd Open-Detector
```

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
```

```bash
.venv/bin/streamlit run app.py
```

**Your own key and style profile.** The Detect page needs neither. The Adversarial Lab generates
challenge data with Gemini: copy `.env.example` to `.env` and put your key after `GEMINI_API_KEY=` (free at
[aistudio.google.com/apikey](https://aistudio.google.com/apikey)). To write in a style, save a
JSON style profile in `data/styles/`; the page lists every JSON file there. To measure one from a
sample of your own writing:

```bash
.venv/bin/python extract_baseline.py my_writing.md --out data/styles/my_profile.json
```

Git ignores both `.env` and `data/`, so your key and your profile stay on your machine.

The app has two pages, linked from the bar at the top. The sidebar picks the detector build both
pages use: **Full** (all three engines, about 3.5 GB) or **Slim** (Desklib alone on the CPU, about
1.5 GB). Machines with 8 GB of RAM or less start on Slim, which scores as well as Full on the
held-out test. Inputs and results stay put when you switch pages.

**Detect** accepts pasted text, `.md` Markdown files, `.docx` Word files and `.pptx` PowerPoint
files. The result shows the verdict and AI likelihood against the two cut-offs, each engine's
reading, and sentence highlights in three colours (likely human, needs review, likely AI, shaded by
how deep into its band each score sits) under a clickable map of the whole text. A chart and a table
show every sentence's score in reading order, and the details include how far each engine pushed the
score and the calibration. An uploaded document gets a full-document score, and each extracted
paragraph, table cell or text box with at least 150 words of prose is also scored on its own and
charted against the cut-offs; shorter sections are listed without a verdict. For a
worksheet, a toggle scores just the filled-in answer cells of its question/answer tables, like the
command line's `--answers-only`. Reference documents can be uploaded alongside to read; their text
is never scored. The detector does not fill templates or rewrite content to target a score.

**Adversarial Lab** generates synthetic, labelled AI challenge samples for detector evaluation.
Enter a test prompt or upload a Markdown, Word or PowerPoint source, then pick a task: a new
challenge sample, a section-level challenge, answers for a test worksheet, or challenges for each
text block while headings, titles, questions and tables stay fixed. If a file doesn't support a task,
the page says why. Document tasks export a
separate Markdown/DOCX/PPTX copy that keeps the uploaded structure; the original is left unchanged.
The two batch tasks run in the background, save each block as it is written, and resume where they
stopped. Short labels, names, dates and numbers ("14 August", "Done", "1") stay as they are: only
prose blocks, a sentence of four words or more or eight words without one, are generated. They
generate several neighbouring blocks in one request (**Blocks per request**, 6 by default, counted in
30-word blocks, so a run of one-line blocks shares a request and a long paragraph takes the room of
several), so the style profile and the document are sent once per group rather than once per block,
and write a few groups at the same time (**Requests at once**, 2 by default). The reply is JSON with
one entry per block; a block it leaves out is generated again on its own.

**How closely to follow the original** decides what a document rewrite keeps of whoever wrote the
original, a person or another model. *Writer's sentences*, the default, gives each block a sentence
plan borrowed from a passage of the style profile's writer about as long as the block (its sentence
lengths and punctuation, never its words), and the model fits the original's meaning into those
sentences in the profile's voice. *Same structure* shows the model each of the block's sentences
numbered with its length and asks for one new sentence per sentence, at about its length and with its
punctuation, then puts the original's paragraph breaks and spacing (double spaces included) back around
the new sentences. That hides AI text in a person's writing, but not in an AI's: an AI original's even
sentences survive any rewrite done in place. Writer's sentences needs a profile with sentence shapes
(the built-in one, or one with the writer's writing beside it, which a public deploy doesn't send);
without them it works as same structure. *Light edit* also keeps most of the original's words. *Free
rewrite* keeps only each block's meaning and length. Each saved block records its **structure match**
(how closely it keeps the original's sentence lengths, punctuation and paragraphs, 0 to 100) and
**words kept** (the share of the original's words it still has). A block that keeps 90% or more of
its original's words is the original handed back, not AI text: it is asked for once more in new
words, and if it still comes back as the original it is left out of the AI-labelled data.

**Rewrite the draft in the writer's sentences**, on by default for a sample written from instructions
in a style profile with sentence shapes, writes the piece and then rewrites it passage by passage in
the profile's voice and in sentence shapes borrowed from its writer, the way a document rewrite does.
Written from instructions alone, a sample keeps the model's own plan of what to say: complete, balanced
and in order, which the detector reads whatever voice the model is told to use. Measured on
2026-10-30 with `python -m eval.tune_prompts`, the restyle took samples from 81-92% AI likelihood
(0-1 of 9 read as human) to the figures in `eval/reports/style_profile_ab.md`. It costs one more
request per sample, and the draft is asked for a quarter longer, since a model fitting points into
borrowed sentences writes short.

**This document is my own writing** also saves each original block, and the whole original, as
human-labelled data (label 0) beside its generated version, under the same `group_key`, so a split
keeps them together. That gives paired data: the same content written by you and by an AI imitating
you, which is what a detector learns the difference from best. Each original is saved once, however
often it is regenerated. Leave it off for a document that is AI-written or someone else's. Every
generated sample also records its thinking level, temperature, style profile and blocks per request. Adaptive challenge mode
generates up to three labelled candidates and uses detector feedback plus the selected style-profile
measurement to rank them. The target, profile and candidate scores are recorded as evaluation
metadata. The current detector target is below 5% AI likelihood; a selected style profile also uses
a 35th-percentile floor when ranking candidates. These are experiment settings, not guarantees.
A style profile picked from `data/styles/` becomes part of the generation condition, and
its file name joins the challenge label.

**Priorities** decide how a run trades four measured goals: lower detector score, style-profile
match, speed and quota. Spend 12 points over them (at most 6 on one) and the page picks the
measured setting with the best points-weighted score, shows it with the time, requests and tokens it
should take, and draws the points as a diamond. Even points are the default. The settings are a
model, its thinking level (how long Gemini reasons before it writes), its temperature and how many
blocks go in one request; speed points also send more requests at once. **Set by hand** chooses them
yourself, creativity (the temperature) included.

The measurements come from a benchmark run on a document:

```bash
.venv/bin/python -m eval.tune_generation report.docx
```

It rewrites the same run of prose blocks (250 words by default) with each setting, the way a batch
does, and records the detector's AI likelihood of each generated challenge, its style match, seconds
per block, and requests and Gemini's token count per block. Temperatures (`--temperatures`, 0.8, 1.0
and 1.3 by default) are measured on the baseline model, and carried to the other models that take a
temperature; Gemini 3.6, 3.7 and 3.8 Flash use their own. Quota also counts each model's free requests
a day: Gemini's quota errors state the limit, so a model that runs out, in the benchmark or in one of
your batches, has it recorded in `data/model_limits.json`, and `--daily-limit MODEL=N` sets one by
hand (AI Studio lists your project's limits). A model that times out twice is skipped for the rest of
the run. The page then weighs how much of that allowance the document in hand would take:
a 100-block batch on a model allowing 20 requests a day uses most of the day. The style match ranks
a generated sample among passages of the writer's own writing (`data/styles/<name>/corpus.jsonl`): the share of them further from the
profile's measurements than it is, so 50% is as typical as the writer's median passage. Measuring
every combination would cost more requests than the free tier gives, so each model is measured at
each thinking level with the largest group, the group sizes on one model, and the rest predicted by
adding those effects (`--design full` measures everything). The run resumes where it stopped and
skips a model whose quota runs out; every rewrite is logged as labelled AI data. It writes
`models/generation_tuning.json` and a report, `eval/reports/tuning.md`, with the spread between
repeats: differences smaller than that are noise.

`--no-style` writes without the style profile while still measuring against it, the control for the
profile's effect. `--structure` sets how closely each rewrite follows its original: `same` (the
default, as on the page), `light` or `free`. The report adds each setting's structure match and the
share of the original's words its rewrites keep. On 2026-10-30, rewriting the writer's own report
without any profile, following its sentences one for one took the detector's AI likelihood from 95-100%
(a free rewrite) down to 3-8%. A free rewrite in the writer's own style turned out to be the original
handed back, which is why an earlier measurement credited the profile with the same drop
(`eval/reports/style_profile_ab.md`).

How much of this the detector catches is measured with

```bash
.venv/bin/python -m eval.redteam_report
```

which scores every saved sample of 150 words or more (`--profile full` for every engine) and writes
`eval/reports/redteam.md`: the share of AI samples it misses and of your originals it wrongly flags,
with Wilson intervals, overall and by attack, model, thinking level, temperature and blocks per
request. It changes nothing and caches its scores, so a second run scores only new samples. The miss
rate is the number to bring down by training on these samples; refitting the combiner alone can only
reweight the engines it already has.

Each sample's probability, verdict, calibration state and detector build are recorded next to
`label: 1` in `data/essays/local_redteam.jsonl`. Standard generation scores a sample once; adaptive
challenge mode also records every candidate score and uses those measurements to select a candidate.
These results measure this detector's responses on labelled synthetic data; they do not establish
performance against every detector or prove authorship. The
latest sample downloads as a `.txt` file, and it and any document copy are also saved under
`data/essays/exports/` with an `AI_REDTEAM_` filename. **Analyze in Detect** sends it to the Detect
page for a full analysis with highlights. The **Saved samples** tab totals every recorded sample:
how many the detector scored, and how often each attack passed as human.

The page uses Gemini's Interactions API with the key in `GEMINI_API_KEY` (the project `.env`), and
tries a configurable fallback list after quota or temporary service errors. Its defaults include
`gemini-3.5-flash-lite`, `gemini-3.6-flash`, `gemini-3.1-flash-lite`, and `gemini-3-flash-preview`.
The older 2.5 Flash models are omitted because Google restricts them for new users. After those it
falls back to a local Ollama model (`gemma3:4b` by default), and the sidebar says whether that can
run. `OPENDETECTOR_REDTEAM=off` leaves the page out, as the Hugging Face deploy does.

On the public Streamlit site the page runs in **public mode** (`OPENDETECTOR_PUBLIC=on`). Each
visitor pastes their own Gemini key in the sidebar. The server sends it to Google in each request's
header and never writes it anywhere, and a batch lets go of it when it ends. By default it lasts only
while the tab is open; **Remember it on this device** asks the visitor's own browser to keep it for
this site (`ui/device_storage.py`, a small component that answers only the app on its own origin),
so it fills in again on that device and no other, and **Forget it on this device** removes it. Samples
are written in the built-in human profile unless the visitor adds their own with **Add your own style
profile**, kept only for that browser session; the host's own keys and `data/styles/` profiles are
never used. Each visitor sees only their own samples and batches, and reloading the page starts
afresh, so download what you want to keep. Public mode always writes with Gemini: the API address
can't be changed and there is no local fallback. The Guide is there too, saving nothing: it explains
where a visitor's key goes.

```bash
.venv/bin/python -m opendetector essay.txt
```

The command line also reads `.md`, `.docx` and `.pptx` files. For a document it scores the whole
text and, separately, every section of at least 150 words. Worksheets are mostly tables of short
cells and often abstain as not prose, so `--answers-only` scores just the answer cells of their
question/answer tables. `--profile slim` runs Desklib alone. In Claude Code, `/ai-detect <file>`
runs the same commands and summarizes the result.

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

**2. Academic writing,** a domain where false positives can affect students. Download PERSUADE 2.0 (human essays
from grades 6-12, with English-learner status) from
[github.com/scrosseye/persuade_corpus_2.0](https://github.com/scrosseye/persuade_corpus_2.0).
Then generate labelled AI challenge samples for the same assignments. The generator defaults to Gemini's Interactions
API, using `GEMINI_API_KEY` from the project's `.env` file (see below):

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

Repeat the generation for other model families and for `--style humanize`, the detector-evasion
condition used to measure robustness against humanized AI outputs. This condition is explicit in
the benchmark, and generated examples remain labelled AI. For other providers, pass `--base-url` and `--api-key-env` (OpenAI,
OpenRouter, or a local LM Studio/Ollama server, which needs no key). Any of your own labelled
data works too: a JSONL with `text` and `label` (`human`/`ai`).

To test the detector against AI text conditioned on one measured writing profile, add `--style-profile` with a
JSON style profile. Its rules and measurements become the system prompt: sentence rhythm,
vocabulary, banned phrases, and mechanical slips such as missing apostrophes. Keep profiles in
`data/styles/`, which git ignores, because a profile of real writing is personal. The profile's
file name joins the attack label (`plain+student_profile`), so the report gives it its own row:

```bash
.venv/bin/python -m eval.generate_essays --persuade persuade_2.0.csv --n 300 --style-profile data/styles/student_profile.json --out data/essays/gemini_profile.jsonl
```

One profile ships with the code: `styles/human_default_profile.json`, the Adversarial Lab's default
("Human (built in)"). It describes no one writer but how people write, measured from 1,750 human texts
by different writers (finished work from news, science abstracts, Wikipedia and stories; quick notes
from argument posts, explanations and reviews) with `python -m eval.build_human_profile`, which keeps
numbers only: the measurements its prose is written from, the sentence shapes Writer's sentences
borrows, and the yardstick its style match ranks against. A profile of one writer, added on the page or
saved in `data/styles/`, reads more like that writer; the built-in one is the control it is compared
with.

A profile in the format of `style_profile_template.json` keeps two registers apart: **finalized**
(finished essays, reports, stories, answers) and **notes** (quick notes written for oneself). Each
habit and slip carries a measured rate and a status: established, emerging, or seen once, which is
used at most once in a long text. Only the chosen register is sent (`--register`, or **Register** on
the Red team page), so quick-note slips never leak into finished work, and the prompt is about a
third shorter. `alexandria_profile` is kept up to date from the writer's Obsidian vault by
`update_style_profile.py` and the `/style-profile` skill. Finished pieces saved in
`data/styles/alexandria/finalized/` are measured into the finalized register: Markdown, text, Word,
PowerPoint or PDF, each counted in full unless `weights.json` there gives it less. Only the writer's
own prose counts: long quotations, verse, equations and tables are left out, and so is any text a blank
template saved in `finalized/templates/` also has, so a filled-in worksheet gives only its answers.
Vault notes count too, by the rung the writer gave them: Gold (finished) at half weight, Silver
(drafted) at a fifth, Bronze and fleeting captures not at all. With finished pieces in the folder, a
finalized-register sample's style match is measured against them rather than against the notes, and
**Show the model the writer's own passages** offers passages from them.

The supplied writer profile is saved as `data/styles/pasted_style_profile.json`. To generate
mirrors in that style, use:

```bash
.venv/bin/python -m eval.generate_essays --persuade persuade_2.0.csv --n 300 --style-profile data/styles/pasted_style_profile.json --out data/essays/gemini_pasted_style.jsonl
```

**API keys** live in a `.env` file in the project root, which git ignores. Copy `.env.example`
to start one. The scripts load it automatically, and a variable already exported in your shell
takes priority:

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

## Install as a Mac app

`deploy/macos/build_app.py` makes **Open Detector.app** in `~/Applications` (`--dest` picks another
folder). It opens from Launchpad, Spotlight, the Finder or the Dock and shows the app in a window
of its own, with the usual menus, and opening it again brings the open window forward. It needs
the Xcode command line tools (`xcode-select --install`) to compile the app's small launcher.

```bash
.venv/bin/python deploy/macos/build_app.py
```

The app runs this checkout with its `.venv`, so a `git pull`, a refit or a new style profile shows
up the next time it opens; build it again only after moving the project folder. When it opens it
starts the server on `127.0.0.1:8517` (another free port if that one is taken), reachable from
this Mac only, and shows a loading page until the server answers. Closing the window or quitting
stops the server. If an Adversarial Lab batch is running, it asks first; the saved blocks resume as
usual. The server's output goes to `~/Library/Logs/Open Detector/server.log`, and if the server
stops the window shows the log with a button to start it again. To run the window without
building the app:

```bash
.venv/bin/python deploy/macos/launcher.py
```

## Deploy free on Streamlit Community Cloud (slim build)

Community Cloud's free tier caps an app at about 2.7 GB of RAM, too little for the full
ensemble. The **slim** profile runs Desklib alone, the strongest single engine, in float32 on
the CPU. It peaks around 1.5 GB, its scores match the full build's Desklib scores (correlation
1.000), and it has its own combiner, `models/combiner_slim.json`, so its verdicts are still
calibrated. The same `app.py` runs both builds; `deploy/streamlit_cloud/streamlit_app.py` just
pins the slim profile and runs the Adversarial Lab in public mode (see above). The theme comes from
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

To run it locally as the public site does:

```bash
OPENDETECTOR_PROFILE=slim OPENDETECTOR_PUBLIC=on .venv/bin/streamlit run app.py
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
`ui` package and theme (without the Adversarial Lab), the `opendetector` package, the fitted
combiner and the files in `deploy/huggingface/`. No `.env` or API keys are uploaded. Run the same
command again to redeploy after changes. Add `--public` for a
public Space, or change its visibility later in the Space's settings. The first analysis after a
restart downloads about 4 GB of models and takes a few minutes.

## Adversarial evaluation

`align_engine.py` is an adaptive detector-evasion experiment. It has Claude generate a text,
scores it with the ensemble, and requests revisions to flagged passages until the AI likelihood
falls under a configured target (default 5%) or it runs out of attempts. This feedback loop is
reported openly as an adversarial test condition; all drafts are labelled AI. It needs Anthropic
credentials (`ANTHROPIC_API_KEY`, or `ant auth login`).

```bash
.venv/bin/python align_engine.py "Write a 600-word essay on loneliness in Of Mice and Men" --log data/attacks/mice.jsonl
```

```bash
.venv/bin/python -m eval.featurize local --path data/attacks/mice.jsonl --out eval/cache/attacks.jsonl
```

Every draft goes into the log labelled as AI. Report the share of prompts that reach the target,
the detector build and the attack settings alongside the result. These are benchmark outcomes for
this detector and test distribution, not universal claims about detector evasion. The resulting
hard positives can support detector training in Phase 2. The harness loads all three engines, so don't run it
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
