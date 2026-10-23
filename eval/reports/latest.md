# Evaluation report (2026-10-23)

Data: eval/cache/raid.jsonl. Dropped 0 texts under 150 words.
Split by source document: fit 848, calibration 266, test 304 (138 human, 166 AI).

## Detectors on the test split

| Detector | AUROC | TPR @ 1% FPR | TPR @ 5% FPR |
|---|---|---|---|
| GLYPH | 0.8883 | 19.3% | 47.0% |
| Desklib | 0.9885 | 86.1% | 91.6% |
| Binoculars | 0.7306 | 48.8% | 53.0% |
| Combined | 0.9878 | 86.1% | 92.2% |

## Deployed thresholds

- **Likely AI** when p > 0.7357 (target FPR 1%): test FPR 2.90% (95% CI 1.13%–7.22%), catches 88.6% of AI texts.
- **Needs review** when p > 0.3153 (target FPR 5%): test FPR 7.25%, catches 96.4% of AI texts.

## Combiner weights (standardized features)

- `desklib_mean`: +2.996
- `desklib_max`: +1.879
- `glyph_mean`: +0.683
- `bino_doc`: -0.630
- `bino_min`: +0.107
- `glyph_max`: -0.099

## By domain (at the Likely-AI threshold)

| domain | human n | FPR (95% CI) | AI n | TPR |
|---|---|---|---|---|
| abstracts | 14 | 0.00% (0.00%–21.53%) | 16 | 100.0% |
| books | 16 | 0.00% (0.00%–19.36%) | 24 | 95.8% |
| news | 34 | 0.00% (0.00%–10.15%) | 24 | 83.3% |
| poetry | 24 | 8.33% (2.32%–25.85%) | 12 | 75.0% |
| recipes | 18 | 0.00% (0.00%–17.59%) | 28 | 85.7% |
| reddit | 8 | 0.00% (0.00%–32.44%) | 23 | 91.3% |
| reviews | 9 | 22.22% (6.32%–54.74%) | 9 | 100.0% |
| wiki | 15 | 0.00% (0.00%–20.39%) | 30 | 83.3% |

## By attack (at the Likely-AI threshold)

| attack | human n | FPR (95% CI) | AI n | TPR |
|---|---|---|---|---|
| alternative_spelling | 9 | 0.00% (0.00%–29.92%) | 7 | 100.0% |
| article_deletion | 0 | – | 11 | 81.8% |
| homoglyph | 25 | 4.00% (0.71%–19.54%) | 14 | 100.0% |
| insert_paragraphs | 19 | 5.26% (0.94%–24.64%) | 21 | 95.2% |
| none | 3 | 0.00% (0.00%–56.15%) | 11 | 100.0% |
| number | 6 | 0.00% (0.00%–39.03%) | 17 | 76.5% |
| paraphrase | 4 | 50.00% (15.00%–85.00%) | 6 | 33.3% |
| perplexity_misspelling | 10 | 0.00% (0.00%–27.75%) | 9 | 100.0% |
| synonym | 7 | 0.00% (0.00%–35.43%) | 13 | 76.9% |
| upper_lower | 19 | 0.00% (0.00%–16.82%) | 11 | 90.9% |
| whitespace | 16 | 0.00% (0.00%–19.36%) | 19 | 78.9% |
| zero_width_space | 20 | 0.00% (0.00%–16.11%) | 27 | 100.0% |

## By model (at the Likely-AI threshold)

| model | human n | FPR (95% CI) | AI n | TPR |
|---|---|---|---|---|
| chatgpt | 0 | – | 12 | 100.0% |
| cohere | 0 | – | 4 | 25.0% |
| cohere-chat | 0 | – | 9 | 55.6% |
| gpt2 | 0 | – | 16 | 93.8% |
| gpt3 | 0 | – | 2 | 50.0% |
| gpt4 | 0 | – | 28 | 85.7% |
| human | 138 | 2.90% (1.13%–7.22%) | 0 | – |
| llama-chat | 0 | – | 23 | 100.0% |
| mistral | 0 | – | 28 | 85.7% |
| mistral-chat | 0 | – | 15 | 100.0% |
| mpt | 0 | – | 24 | 91.7% |
| mpt-chat | 0 | – | 5 | 100.0% |
