# Open Detector

An AI-text detector with a Streamlit app of two pages:

- **Detect** scores pasted text or a Word, PowerPoint or Markdown file. It gives a verdict (*Likely
  AI*, *Needs review* or *Likely human*), a calibrated AI likelihood, each engine's reading,
  sentence highlights and a score for every long section. For a worksheet it can score just the
  filled-in answer cells.
- **Red team** writes labelled AI samples to test the detector with: a new sample from a prompt,
  a rewrite of one section of an uploaded document, a new answer in every answer box, or a rewrite
  of every text block. It writes with Gemini, can follow a writer's style profile, scores each
  sample with the detector and saves it as labelled AI data.

The live demo runs the detector only: https://open-detector.streamlit.app

## How it scores

Three engines score overlapping windows of the whole text: GLYPH (DeBERTa-v3-base), Desklib
(DeBERTa-v3-large) and Binoculars (a Qwen2.5-0.5B pair). A logistic-regression combiner turns
their readings into one AI likelihood, with cut-offs set on held-out human writing for a 1%
(*Likely AI*) and 5% (*Needs review*) false-positive rate. It gives no verdict on texts under 150
words or on non-prose such as poems and lists, and it strips hidden characters and look-alike
letters before scoring. The *Slim* build runs Desklib alone on the CPU in about 1.5 GB.

## Run it yourself

```bash
git clone -b feat/phase1-ensemble https://github.com/lukaberadze1/Open-Detector.git && cd Open-Detector
```

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
```

```bash
cp .env.example .env
```

Put your Gemini key after `GEMINI_API_KEY=` in `.env` (free at
[aistudio.google.com/apikey](https://aistudio.google.com/apikey)). Only the Red team page needs it.
To write in a style, save a JSON style profile in `data/styles/`; the page lists every one there.
Git ignores `.env` and `data/`, so your key and profile stay on your machine.

```bash
.venv/bin/streamlit run app.py
```

The app opens in your browser with **Detect** and **Red team** in the top bar. The first run
downloads the models: 1.7 GB for Slim, about 4.4 GB for every engine. Machines with 8 GB of RAM or
less start on Slim.

To score a file from the command line:

```bash
.venv/bin/python -m opendetector essay.docx
```

## More

- [DETECTOR.md](DETECTOR.md): setup, the evaluation and calibration workflow, deployment and
  red-teaming.
- [eval/reports/latest.md](eval/reports/latest.md): the current evaluation results.
- Known limits: English only; AI paraphrasing of human writing is the hardest case; a score is
  statistical evidence, not proof of who wrote a text.

This project started as a fork of followsci's Open Detector, a single BERT classifier. Its original
README is kept in [README_EN.md](README_EN.md) and [README_ZH.md](README_ZH.md).
