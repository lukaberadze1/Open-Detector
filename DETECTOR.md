# Open Detector: ensemble detector and eval harness

```
text ─► normalize ─► sentences ─► overlapping windows ─┬─► GLYPH (DeBERTa-v3-base)   ─┐
        (zero-width,  (spans)     doc: ~350 words      ├─► Desklib (DeBERTa-v3-large) ├─► combiner ─► calibrated p ─► verdict
         homoglyphs)              local: ~100 words    └─► Binoculars (Qwen pair)     ─┘  (logistic    (thresholds set on
                                                                                         regression)  held-out human text)
```

- **Whole document, never truncated.** Text is cut into overlapping windows of whole sentences,
  and every window is scored. Sentences inherit the average score of the windows around them,
  so no sentence is ever judged alone.
- **Binoculars replaces raw perplexity.** Raw perplexity is the signal that flags simple prose
  and non-native writers. Binoculars normalizes it by how predictable *any* text looks to a
  second model.
- **A learned combiner replaces hand-tuned offsets.** It is a logistic regression over engine
  features, fitted on labelled data. The *Likely AI* threshold is set on held-out human texts
  for a 1% false-positive rate, and *Needs review* for 5%, using a conformal quantile that
  carries a finite-sample guarantee.
- **Evasion tricks are removed and reported.** Zero-width characters and Cyrillic/Greek
  look-alike letters are stripped before scoring, and the app says how many it found.
- **It abstains on short text** (under 150 words; Turnitin reportedly needs about 300).

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
document, because attacked copies of one text aren't independent errors. RAID's "human" texts
under its paraphrase attack are left out: an AI paraphraser rewrote them.

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

## Deploy to Hugging Face Spaces

The app runs on a free CPU Space: 16 GB of RAM covers the models in full precision. Sign in
once with a **write** token from huggingface.co/settings/tokens, then deploy:

```bash
.venv/bin/hf auth login
```

```bash
.venv/bin/python deploy/huggingface/deploy.py
```

This creates a private Space, `<your-username>/open-detector`, and uploads only `app.py`, the
`opendetector` package, the fitted combiner and the files in `deploy/huggingface/`. No `.env` or
API keys are uploaded. Run the same command again to redeploy after changes. Add `--public` for a
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
  look; only the document score is calibrated. A sentence on the edge of an AI-written passage
  inherits some of its score, because its windows overlap the passage. Fixing this needs a
  boundary-aware model (Phase 3).
- In testing on an 8 GB Mac, Streamlit crashed once (segfault) under heavy memory pressure
  with all three engines loaded. It didn't recur in later runs. If it happens, switch off
  Binoculars in the sidebar.
- The 0.5B Binoculars pair is weaker than the paper's 7B pair.
- Commercial humanizers remain the hardest case. The way to fix that is training a model on
  humanized text, which is Phase 2 of the roadmap.
- A score is statistical evidence, not proof. Don't use it as the only basis for an academic
  integrity decision.
