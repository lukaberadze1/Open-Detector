# Evaluation report (2026-10-23)

Data: eval/cache/raid.jsonl. Dropped 0 texts under 150 words and 25 RAID "human" texts rewritten by its AI paraphraser.
Split by source document: development 1093 (480 human, thresholds set on 5-fold out-of-fold scores), test 300 (134 human texts from 125 source documents, 166 AI).

## Detectors on the test split

| Detector | AUROC | TPR @ 1% FPR | TPR @ 5% FPR |
|---|---|---|---|
| GLYPH | 0.8886 | 19.3% | 47.0% |
| Desklib | 0.9900 | 86.1% | 94.6% |
| Binoculars | 0.7375 | 50.6% | 56.6% |
| Combined | 0.9903 | 87.3% | 95.2% |

## Deployed thresholds

False-positive rates count human source documents (flagged if any version is); per-text rates follow.

- **Likely AI** when p > 0.8347 (target FPR 1%): flags 1/125 human documents, 0.80% (95% CI 0.14%–4.39%); per text 2/134, 1.49%. Catches 88.6% of AI texts.
- **Needs review** when p > 0.3547 (target FPR 5%): flags 8/125 human documents, 6.40% (95% CI 3.28%–12.12%). Catches 96.4% of AI texts. (above target, but within sampling error)

## Combiner weights (standardized features)

- `desklib_mean`: +3.163
- `desklib_max`: +1.957
- `bino_doc`: -0.836
- `style_aiisms`: +0.683
- `glyph_mean`: +0.645
- `glyph_max`: -0.201
- `bino_min`: +0.134
- `style_mattr`: +0.089
- `style_len_cv`: -0.083

## AI-paraphrased category

Fitted on 82 rewritten and 58 as-written development AI texts. Called paraphrased when p > 0.9294, which lets at most 5% of as-written AI texts through as paraphrased.

On test AI texts: AUROC 0.8565; catches 36.8% of 19 rewritten texts and calls 0.0% of 11 as-written texts paraphrased.

| attack | AI n | called paraphrased |
|---|---|---|
| alternative_spelling | 7 | 14.3% |
| article_deletion | 11 | 18.2% |
| homoglyph | 14 | 7.1% |
| insert_paragraphs | 21 | 4.8% |
| none | 11 | 0.0% |
| number | 17 | 11.8% |
| paraphrase | 6 | 66.7% |
| perplexity_misspelling | 9 | 11.1% |
| synonym | 13 | 23.1% |
| upper_lower | 11 | 18.2% |
| whitespace | 19 | 26.3% |
| zero_width_space | 27 | 3.7% |

## By domain (at the Likely-AI threshold)

| domain | human n | FPR (95% CI) | AI n | TPR |
|---|---|---|---|---|
| abstracts | 14 | 0.00% (0.00%–21.53%) | 16 | 100.0% |
| books | 16 | 0.00% (0.00%–19.36%) | 24 | 95.8% |
| news | 34 | 0.00% (0.00%–10.15%) | 24 | 83.3% |
| poetry | 24 | 8.33% (2.32%–25.85%) | 12 | 91.7% |
| recipes | 18 | 0.00% (0.00%–17.59%) | 28 | 85.7% |
| reddit | 8 | 0.00% (0.00%–32.44%) | 23 | 87.0% |
| reviews | 5 | 0.00% (0.00%–43.45%) | 9 | 100.0% |
| wiki | 15 | 0.00% (0.00%–20.39%) | 30 | 80.0% |

## By attack (at the Likely-AI threshold)

| attack | human n | FPR (95% CI) | AI n | TPR |
|---|---|---|---|---|
| alternative_spelling | 9 | 0.00% (0.00%–29.92%) | 7 | 85.7% |
| article_deletion | 0 | – | 11 | 81.8% |
| homoglyph | 25 | 4.00% (0.71%–19.54%) | 14 | 100.0% |
| insert_paragraphs | 19 | 5.26% (0.94%–24.64%) | 21 | 95.2% |
| none | 3 | 0.00% (0.00%–56.15%) | 11 | 100.0% |
| number | 6 | 0.00% (0.00%–39.03%) | 17 | 76.5% |
| paraphrase | 0 | – | 6 | 50.0% |
| perplexity_misspelling | 10 | 0.00% (0.00%–27.75%) | 9 | 100.0% |
| synonym | 7 | 0.00% (0.00%–35.43%) | 13 | 76.9% |
| upper_lower | 19 | 0.00% (0.00%–16.82%) | 11 | 90.9% |
| whitespace | 16 | 0.00% (0.00%–19.36%) | 19 | 84.2% |
| zero_width_space | 20 | 0.00% (0.00%–16.11%) | 27 | 96.3% |

## By model (at the Likely-AI threshold)

| model | human n | FPR (95% CI) | AI n | TPR |
|---|---|---|---|---|
| chatgpt | 0 | – | 12 | 100.0% |
| cohere | 0 | – | 4 | 25.0% |
| cohere-chat | 0 | – | 9 | 55.6% |
| gpt2 | 0 | – | 16 | 93.8% |
| gpt3 | 0 | – | 2 | 50.0% |
| gpt4 | 0 | – | 28 | 85.7% |
| human | 134 | 1.49% (0.41%–5.28%) | 0 | – |
| llama-chat | 0 | – | 23 | 100.0% |
| mistral | 0 | – | 28 | 85.7% |
| mistral-chat | 0 | – | 15 | 100.0% |
| mpt | 0 | – | 24 | 91.7% |
| mpt-chat | 0 | – | 5 | 100.0% |
