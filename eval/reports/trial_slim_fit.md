# Evaluation report (2026-11-07)

Data: eval/cache/raid.jsonl, eval/cache/mage.jsonl, /Users/luka2/Open-Detector/eval/cache/daigt.jsonl. Dropped 0 texts under 150 words and 302 that aren't prose (poetry, lists), which the detector doesn't score; counted 25 RAID "human" texts rewritten by its AI paraphraser as AI.
Split by source document: development 3824 (1799 human, thresholds set on 5-fold out-of-fold scores), test 976 (441 human texts from 438 source documents, 535 AI).

## Detectors on the test split

| Detector | AUROC | TPR @ 1% FPR | TPR @ 5% FPR |
|---|---|---|---|
| GLYPH | 0.8310 | 14.7% | 50.3% |
| Desklib | 0.9916 | 92.0% | 97.0% |
| Binoculars | 0.6775 | 38.8% | 46.5% |
| Burstiness | 0.6912 | 3.1% | 17.5% |
| Combined | 0.9915 | 92.5% | 97.0% |

## Deployed thresholds

False-positive rates count human source documents (flagged if any version is); per-text rates follow.

- **Likely AI** when p > 0.7477 (target FPR 1%): flags 6/438 human documents, 1.37% (95% CI 0.63%–2.96%); per text 6/441, 1.36%. Catches 92.5% of AI texts. (above target, but within sampling error)
- **Needs review** when p > 0.1500 (a fixed cut-off, not calibrated): flags 45/438 human documents, 10.27% (95% CI 7.77%–13.47%). Catches 98.5% of AI texts.

## Combiner weights (standardized features)

- `desklib_mean`: +4.172
- `desklib_max`: +1.109

## By dataset (at the Likely-AI threshold)

| dataset | human n | FPR (95% CI) | AI n | TPR |
|---|---|---|---|---|
| daigt | 188 | 0.53% (0.09%–2.95%) | 223 | 100.0% |
| mage | 155 | 3.23% (1.39%–7.33%) | 177 | 89.8% |
| raid | 98 | 0.00% (0.00%–3.77%) | 135 | 83.7% |

## By domain (at the Likely-AI threshold)

| domain | human n | FPR (95% CI) | AI n | TPR |
|---|---|---|---|---|
| "A Cowboy Who Rode the Waves" | 10 | 0.00% (0.00%–27.75%) | 13 | 100.0% |
| Car-free cities | 20 | 0.00% (0.00%–16.11%) | 41 | 100.0% |
| Cell phones at school | 7 | 0.00% (0.00%–35.43%) | 10 | 100.0% |
| Community service | 6 | 0.00% (0.00%–39.03%) | 5 | 100.0% |
| Distance learning | 19 | 0.00% (0.00%–16.82%) | 28 | 100.0% |
| Does the electoral college work? | 14 | 0.00% (0.00%–21.53%) | 37 | 100.0% |
| Driverless cars | 22 | 0.00% (0.00%–14.87%) | 5 | 100.0% |
| Exploring Venus | 8 | 0.00% (0.00%–32.44%) | 7 | 100.0% |
| Facial action coding system | 18 | 0.00% (0.00%–17.59%) | 12 | 100.0% |
| Grades for extracurricular activities | 9 | 0.00% (0.00%–29.92%) | 5 | 100.0% |
| Mandatory extracurricular activities | 8 | 0.00% (0.00%–32.44%) | 11 | 100.0% |
| Phones and driving | 12 | 0.00% (0.00%–24.25%) | 6 | 100.0% |
| Seeking multiple opinions | 10 | 10.00% (1.79%–40.42%) | 29 | 100.0% |
| Summer projects | 15 | 0.00% (0.00%–20.39%) | 6 | 100.0% |
| The Face on Mars | 10 | 0.00% (0.00%–27.75%) | 8 | 100.0% |
| abstracts | 14 | 0.00% (0.00%–21.53%) | 15 | 100.0% |
| books | 16 | 0.00% (0.00%–19.36%) | 24 | 95.8% |
| cmv | 21 | 0.00% (0.00%–15.46%) | 14 | 92.9% |
| cnn | 7 | 0.00% (0.00%–35.43%) | 8 | 50.0% |
| dialogsum | 5 | 0.00% (0.00%–43.45%) | 2 | 50.0% |
| eli5 | 18 | 0.00% (0.00%–17.59%) | 11 | 90.9% |
| hswag | 0 | – | 8 | 87.5% |
| imdb | 2 | 0.00% (0.00%–65.76%) | 7 | 71.4% |
| news | 34 | 0.00% (0.00%–10.15%) | 24 | 83.3% |
| poetry | 5 | 0.00% (0.00%–43.45%) | 4 | 50.0% |
| pubmed | 0 | – | 5 | 40.0% |
| recipes | 2 | 0.00% (0.00%–65.76%) | 8 | 62.5% |
| reddit | 8 | 0.00% (0.00%–32.44%) | 20 | 85.0% |
| reviews | 5 | 0.00% (0.00%–43.45%) | 13 | 69.2% |
| roct | 0 | – | 22 | 100.0% |
| sci | 20 | 5.00% (0.89%–23.61%) | 16 | 93.8% |
| squad | 6 | 0.00% (0.00%–39.03%) | 21 | 85.7% |
| tldr | 0 | – | 17 | 100.0% |
| wiki | 14 | 0.00% (0.00%–21.53%) | 27 | 81.5% |
| wp | 33 | 0.00% (0.00%–10.43%) | 22 | 100.0% |
| xsum | 35 | 11.43% (4.54%–25.95%) | 15 | 93.3% |
| yelp | 8 | 0.00% (0.00%–32.44%) | 9 | 100.0% |

## By attack (at the Likely-AI threshold)

| attack | human n | FPR (95% CI) | AI n | TPR |
|---|---|---|---|---|
| alternative_spelling | 9 | 0.00% (0.00%–29.92%) | 7 | 85.7% |
| article_deletion | 0 | – | 7 | 71.4% |
| homoglyph | 14 | 0.00% (0.00%–21.53%) | 9 | 100.0% |
| insert_paragraphs | 8 | 0.00% (0.00%–32.44%) | 18 | 94.4% |
| none | 346 | 1.73% (0.80%–3.73%) | 389 | 97.7% |
| number | 0 | – | 12 | 66.7% |
| paraphrase | 0 | – | 29 | 41.4% |
| perplexity_misspelling | 10 | 0.00% (0.00%–27.75%) | 6 | 100.0% |
| synonym | 7 | 0.00% (0.00%–35.43%) | 8 | 87.5% |
| upper_lower | 19 | 0.00% (0.00%–16.82%) | 11 | 90.9% |
| whitespace | 8 | 0.00% (0.00%–32.44%) | 12 | 75.0% |
| zero_width_space | 20 | 0.00% (0.00%–16.11%) | 27 | 96.3% |

## By model (at the Likely-AI threshold)

| model | human n | FPR (95% CI) | AI n | TPR |
|---|---|---|---|---|
| NousResearch/Llama-2-7b-chat-hf | 0 | – | 15 | 100.0% |
| chat_gpt_moth | 0 | – | 19 | 100.0% |
| chatgpt | 0 | – | 11 | 100.0% |
| cmv_machine_continuation_13B | 0 | – | 1 | 100.0% |
| cmv_machine_continuation_30B | 0 | – | 1 | 0.0% |
| cmv_machine_continuation_GLM130B | 0 | – | 2 | 100.0% |
| cmv_machine_continuation_bloom_7b | 0 | – | 3 | 100.0% |
| cmv_machine_continuation_flan_t5_xxl | 0 | – | 1 | 100.0% |
| cmv_machine_continuation_gpt_j | 0 | – | 1 | 100.0% |
| cmv_machine_continuation_opt_2.7b | 0 | – | 3 | 100.0% |
| cmv_machine_continuation_opt_350m | 0 | – | 1 | 100.0% |
| cmv_machine_continuation_text-davinci-003 | 0 | – | 1 | 100.0% |
| cnn_gpt4 | 0 | – | 2 | 50.0% |
| cnn_gpt4_para | 0 | – | 3 | 66.7% |
| cohere | 0 | – | 3 | 0.0% |
| cohere-chat | 0 | – | 7 | 42.9% |
| cohere-command | 0 | – | 16 | 100.0% |
| darragh_claude_v6 | 0 | – | 19 | 100.0% |
| darragh_claude_v7 | 0 | – | 14 | 100.0% |
| dialogsum_gpt4_para | 0 | – | 1 | 100.0% |
| eli5_machine_continuation_13B | 0 | – | 4 | 100.0% |
| eli5_machine_continuation_30B | 0 | – | 1 | 100.0% |
| eli5_machine_continuation_65B | 0 | – | 1 | 100.0% |
| eli5_machine_continuation_GLM130B | 0 | – | 1 | 100.0% |
| eli5_machine_continuation_gpt_neox | 0 | – | 1 | 100.0% |
| eli5_machine_continuation_opt_1.3b | 0 | – | 1 | 100.0% |
| eli5_machine_continuation_opt_iml_30b | 0 | – | 1 | 100.0% |
| eli5_machine_continuation_text-davinci-002 | 0 | – | 1 | 0.0% |
| falcon_180b_v1 | 0 | – | 19 | 100.0% |
| gpt2 | 0 | – | 13 | 92.3% |
| gpt3 | 0 | – | 2 | 50.0% |
| gpt4 | 0 | – | 24 | 79.2% |
| hswag_machine_continuation_13B | 0 | – | 1 | 100.0% |
| hswag_machine_continuation_GLM130B | 0 | – | 1 | 100.0% |
| hswag_machine_continuation_gpt_j | 0 | – | 1 | 100.0% |
| hswag_machine_continuation_opt_1.3b | 0 | – | 2 | 100.0% |
| hswag_machine_continuation_opt_125m | 0 | – | 1 | 100.0% |
| hswag_machine_continuation_opt_6.7b | 0 | – | 1 | 100.0% |
| hswag_machine_continuation_text-davinci-002 | 0 | – | 1 | 0.0% |
| human | 441 | 1.36% (0.62%–2.94%) | 4 | 0.0% |
| imdb_gpt4 | 0 | – | 1 | 100.0% |
| imdb_gpt4_para | 0 | – | 3 | 100.0% |
| kingki19_palm | 0 | – | 12 | 100.0% |
| llama-chat | 0 | – | 16 | 100.0% |
| llama2_chat | 0 | – | 12 | 100.0% |
| llama_70b_v1 | 0 | – | 16 | 100.0% |
| mistral | 0 | – | 23 | 82.6% |
| mistral-chat | 0 | – | 9 | 100.0% |
| mistral7binstruct_v1 | 0 | – | 14 | 100.0% |
| mistral7binstruct_v2 | 0 | – | 15 | 100.0% |
| mistralai/Mistral-7B-Instruct-v0.1 | 0 | – | 9 | 100.0% |
| mpt | 0 | – | 21 | 100.0% |
| mpt-chat | 0 | – | 2 | 100.0% |
| palm-text-bison1 | 0 | – | 14 | 100.0% |
| paraphraser | 0 | – | 9 | 22.2% |
| pubmed_gpt4_para | 0 | – | 3 | 66.7% |
| radek_500 | 0 | – | 14 | 100.0% |
| radekgpt4 | 0 | – | 15 | 100.0% |
| roct_machine_continuation_13B | 0 | – | 1 | 100.0% |
| roct_machine_continuation_30B | 0 | – | 2 | 100.0% |
| roct_machine_continuation_65B | 0 | – | 1 | 100.0% |
| roct_machine_continuation_GLM130B | 0 | – | 1 | 100.0% |
| roct_machine_continuation_bloom_7b | 0 | – | 2 | 100.0% |
| roct_machine_continuation_opt_1.3b | 0 | – | 1 | 100.0% |
| roct_machine_continuation_opt_125m | 0 | – | 2 | 100.0% |
| roct_machine_continuation_opt_2.7b | 0 | – | 5 | 100.0% |
| roct_machine_continuation_opt_30b | 0 | – | 1 | 100.0% |
| roct_machine_continuation_opt_350m | 0 | – | 1 | 100.0% |
| roct_machine_continuation_opt_6.7b | 0 | – | 1 | 100.0% |
| roct_machine_continuation_opt_iml_max_1.3b | 0 | – | 1 | 100.0% |
| roct_machine_specified_text-davinci-003 | 0 | – | 1 | 100.0% |
| roct_machine_topical_gpt-3.5-trubo | 0 | – | 1 | 100.0% |
| roct_machine_topical_text-davinci-003 | 0 | – | 1 | 100.0% |
| sci_gen_machine_continuation_30B | 0 | – | 1 | 0.0% |
| sci_gen_machine_continuation_7B | 0 | – | 1 | 100.0% |
| sci_gen_machine_continuation_GLM130B | 0 | – | 1 | 100.0% |
| sci_gen_machine_continuation_bloom_7b | 0 | – | 2 | 100.0% |
| sci_gen_machine_continuation_gpt-3.5-trubo | 0 | – | 1 | 100.0% |
| sci_gen_machine_continuation_gpt_neox | 0 | – | 1 | 100.0% |
| sci_gen_machine_continuation_opt_1.3b | 0 | – | 1 | 100.0% |
| sci_gen_machine_continuation_opt_30b | 0 | – | 4 | 100.0% |
| sci_gen_machine_continuation_opt_350m | 0 | – | 1 | 100.0% |
| sci_gen_machine_continuation_opt_iml_max_1.3b | 0 | – | 2 | 100.0% |
| sci_gen_machine_continuation_text-davinci-003 | 0 | – | 1 | 100.0% |
| squad_machine_continuation_13B | 0 | – | 3 | 66.7% |
| squad_machine_continuation_30B | 0 | – | 1 | 0.0% |
| squad_machine_continuation_7B | 0 | – | 2 | 100.0% |
| squad_machine_continuation_flan_t5_xxl | 0 | – | 1 | 100.0% |
| squad_machine_continuation_gpt_j | 0 | – | 1 | 100.0% |
| squad_machine_continuation_gpt_neox | 0 | – | 1 | 100.0% |
| squad_machine_continuation_opt_1.3b | 0 | – | 2 | 100.0% |
| squad_machine_continuation_opt_125m | 0 | – | 1 | 100.0% |
| squad_machine_continuation_opt_2.7b | 0 | – | 2 | 100.0% |
| squad_machine_continuation_opt_30b | 0 | – | 1 | 100.0% |
| squad_machine_continuation_opt_6.7b | 0 | – | 2 | 100.0% |
| squad_machine_continuation_opt_iml_30b | 0 | – | 1 | 100.0% |
| squad_machine_continuation_opt_iml_max_1.3b | 0 | – | 2 | 100.0% |
| squad_machine_continuation_t0_3b | 0 | – | 1 | 0.0% |
| tldr_machine_continuation_13B | 0 | – | 3 | 100.0% |
| tldr_machine_continuation_65B | 0 | – | 1 | 100.0% |
| tldr_machine_continuation_7B | 0 | – | 1 | 100.0% |
| tldr_machine_continuation_GLM130B | 0 | – | 1 | 100.0% |
| tldr_machine_continuation_bloom_7b | 0 | – | 1 | 100.0% |
| tldr_machine_continuation_flan_t5_xxl | 0 | – | 1 | 100.0% |
| tldr_machine_continuation_gpt_j | 0 | – | 1 | 100.0% |
| tldr_machine_continuation_opt_1.3b | 0 | – | 1 | 100.0% |
| tldr_machine_continuation_opt_2.7b | 0 | – | 3 | 100.0% |
| tldr_machine_continuation_opt_30b | 0 | – | 1 | 100.0% |
| tldr_machine_continuation_opt_iml_30b | 0 | – | 1 | 100.0% |
| tldr_machine_continuation_text-davinci-002 | 0 | – | 1 | 100.0% |
| tldr_machine_specified_gpt-3.5-trubo | 0 | – | 1 | 100.0% |
| wp_machine_continuation_13B | 0 | – | 1 | 100.0% |
| wp_machine_continuation_30B | 0 | – | 1 | 100.0% |
| wp_machine_continuation_65B | 0 | – | 2 | 100.0% |
| wp_machine_continuation_GLM130B | 0 | – | 1 | 100.0% |
| wp_machine_continuation_bloom_7b | 0 | – | 1 | 100.0% |
| wp_machine_continuation_gpt-3.5-trubo | 0 | – | 2 | 100.0% |
| wp_machine_continuation_gpt_j | 0 | – | 1 | 100.0% |
| wp_machine_continuation_opt_13b | 0 | – | 1 | 100.0% |
| wp_machine_continuation_opt_350m | 0 | – | 1 | 100.0% |
| wp_machine_continuation_opt_iml_30b | 0 | – | 2 | 100.0% |
| wp_machine_continuation_t0_11b | 0 | – | 1 | 100.0% |
| wp_machine_continuation_text-davinci-002 | 0 | – | 1 | 100.0% |
| wp_machine_continuation_text-davinci-003 | 0 | – | 1 | 100.0% |
| wp_machine_specified_text-davinci-002 | 0 | – | 1 | 100.0% |
| wp_machine_specified_text-davinci-003 | 0 | – | 2 | 100.0% |
| wp_machine_topical_gpt-3.5-trubo | 0 | – | 1 | 100.0% |
| wp_machine_topical_text-davinci-002 | 0 | – | 2 | 100.0% |
| xsum_machine_continuation_13B | 0 | – | 1 | 100.0% |
| xsum_machine_continuation_30B | 0 | – | 1 | 100.0% |
| xsum_machine_continuation_7B | 0 | – | 1 | 100.0% |
| xsum_machine_continuation_opt_350m | 0 | – | 1 | 100.0% |
| xsum_machine_continuation_opt_iml_30b | 0 | – | 1 | 100.0% |
| xsum_machine_continuation_opt_iml_max_1.3b | 0 | – | 1 | 100.0% |
| xsum_machine_continuation_text-davinci-003 | 0 | – | 2 | 100.0% |
| xsum_machine_specified_gpt-3.5-trubo | 0 | – | 1 | 100.0% |
| xsum_machine_specified_text-davinci-002 | 0 | – | 1 | 100.0% |
| xsum_machine_specified_text-davinci-003 | 0 | – | 2 | 100.0% |
| xsum_machine_topical_gpt-3.5-trubo | 0 | – | 1 | 100.0% |
| xsum_machine_topical_text-davinci-002 | 0 | – | 1 | 0.0% |
| xsum_machine_topical_text-davinci-003 | 0 | – | 1 | 100.0% |
| yelp_machine_continuation_30B | 0 | – | 2 | 100.0% |
| yelp_machine_continuation_65B | 0 | – | 3 | 100.0% |
| yelp_machine_continuation_GLM130B | 0 | – | 1 | 100.0% |
| yelp_machine_continuation_bloom_7b | 0 | – | 1 | 100.0% |
| yelp_machine_continuation_gpt_j | 0 | – | 1 | 100.0% |
| yelp_machine_continuation_opt_iml_max_1.3b | 0 | – | 1 | 100.0% |

## By group (at the Likely-AI threshold)

| group | human n | FPR (95% CI) | AI n | TPR |
|---|---|---|---|---|
| NousResearch/Llama-2-7b-chat-hf | 0 | – | 15 | 100.0% |
| all | 253 | 1.98% (0.85%–4.54%) | 312 | 87.2% |
| chat_gpt_moth | 0 | – | 19 | 100.0% |
| cohere-command | 0 | – | 16 | 100.0% |
| darragh_claude_v6 | 0 | – | 19 | 100.0% |
| darragh_claude_v7 | 0 | – | 14 | 100.0% |
| falcon_180b_v1 | 0 | – | 19 | 100.0% |
| kingki19_palm | 0 | – | 12 | 100.0% |
| llama2_chat | 0 | – | 12 | 100.0% |
| llama_70b_v1 | 0 | – | 16 | 100.0% |
| mistral7binstruct_v1 | 0 | – | 14 | 100.0% |
| mistral7binstruct_v2 | 0 | – | 15 | 100.0% |
| mistralai/Mistral-7B-Instruct-v0.1 | 0 | – | 9 | 100.0% |
| palm-text-bison1 | 0 | – | 14 | 100.0% |
| persuade_corpus | 178 | 0.56% (0.10%–3.11%) | 0 | – |
| radek_500 | 0 | – | 14 | 100.0% |
| radekgpt4 | 0 | – | 15 | 100.0% |
| train_essays | 10 | 0.00% (0.00%–27.75%) | 0 | – |
