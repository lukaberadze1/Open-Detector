---
title: Open Detector
emoji: 🔎
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 8501
pinned: false
short_description: Calibrated ensemble AI-text detector
---

# Open Detector

Paste a text of 150+ words to estimate whether it was written by AI. Three engines score it:

- GLYPH, a DeBERTa-v3-base classifier
- Desklib, a DeBERTa-v3-large classifier
- Binoculars, a zero-shot detector built on two small Qwen models

Each engine scores overlapping windows of the whole text, and a calibrated logistic combiner
turns the scores into one AI likelihood. The verdict thresholds are set for a 1% ("Likely AI")
and 5% ("Needs review") false-positive rate. On a held-out RAID test split, the Likely-AI band
flagged 1 of 125 human documents and caught 87% of AI texts.

The first analysis after the Space starts downloads about 4 GB of models, so it can take a few
minutes. Later ones take well under a minute on CPU.

A score is statistical evidence, not proof of authorship. Don't use it as the only basis for
an academic integrity decision.
