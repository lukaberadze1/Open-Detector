# Red-team samples against the detector

247 AI samples and 3 human ones of 150 words or more, scored with the slim detector and its shipped thresholds. Written by `python -m eval.redteam_report`. Brackets are Wilson 95% intervals.

## AI samples

- Caught (Likely AI): 43% (37% to 49%)
- Needs review: 19% (15% to 24%)
- **Missed (Likely human): 38% (32% to 44%)**
- Mean AI likelihood: 57%

### By attack

| Attack | Samples | Missed | Caught | Mean AI likelihood |
|---|---:|---:|---:|---:|
| full_document+alexandria_profile:finalized+writer+t0.8+as-page | 20 | 20% (8% to 42%) | 65% (43% to 82%) | 72% |
| full_document+alexandria_profile:finalized+writer+t0.8+as-page+revise4-sentences | 20 | 65% (43% to 82%) | 15% (5% to 36%) | 33% |
| prompt+human_default_profile:finalized+task+restyle | 18 | 50% (29% to 71%) | 17% (6% to 39%) | 42% |
| full_document+plain | 12 | 0% (0% to 24%) | 100% (76% to 100%) | 98% |
| prompt+alexandria_profile:notes+task+restyle | 12 | 92% (65% to 99%) | 0% (0% to 24%) | 19% |
| prompt+human_default_profile:finalized+task+restyle+persona | 10 | 10% (2% to 40%) | 70% (40% to 89%) | 83% |
| prompt+alexandria_profile:finalized+challenge+old-voice | 9 | 11% (2% to 44%) | 78% (45% to 94%) | 86% |
| prompt+alexandria_profile:finalized+task | 9 | 0% (0% to 30%) | 89% (56% to 98%) | 91% |
| prompt+alexandria_profile:finalized+task+plan | 9 | 0% (0% to 30%) | 89% (56% to 98%) | 92% |
| prompt+human_default_profile:finalized+challenge+old-voice | 9 | 0% (0% to 30%) | 56% (27% to 81%) | 81% |
| prompt+alexandria_profile:finalized+task+restyle | 9 | 22% (6% to 55%) | 33% (12% to 65%) | 57% |
| prompt+alexandria_profile:notes+task | 9 | 22% (6% to 55%) | 22% (6% to 55%) | 61% |
| prompt+human_default_profile:notes+task+restyle | 9 | 33% (12% to 65%) | 33% (12% to 65%) | 55% |
| prompt+alexandria_profile:finalized+task+restyle+revise3 | 9 | 78% (45% to 94%) | 0% (0% to 30%) | 21% |
| prompt+alexandria_profile:notes+task+restyle+revise3 | 9 | 89% (56% to 98%) | 0% (0% to 30%) | 18% |
| prompt+human_default_profile:notes+task | 8 | 0% (0% to 32%) | 88% (53% to 98%) | 92% |
| full_document+alexandria_profile:finalized+writer | 6 | 0% (0% to 39%) | 67% (30% to 90%) | 81% |
| full_document+alexandria_profile:finalized+writer+revise3-sentences | 6 | 100% (61% to 100%) | 0% (0% to 39%) | 7% |
| full_document+alexandria_profile:finalized+writer+revise3-windows | 6 | 100% (61% to 100%) | 0% (0% to 39%) | 7% |
| prompt+human_default_profile:finalized+task | 6 | 0% (0% to 39%) | 100% (61% to 100%) | 98% |
| full_document+alexandria_profile:finalized+writer+t1 | 4 | 25% (5% to 70%) | 75% (30% to 95%) | 70% |
| full_document+alexandria_profile:finalized+writer+t1+revise3-sentences | 4 | 100% (51% to 100%) | 0% (0% to 49%) | 7% |
| prompt+alexandria_profile | 3 | 0% (0% to 56%) | 100% (44% to 100%) | 95% |
| prompt+alexandria_profile:finalized+task+restyle+examples | 3 | 0% (0% to 56%) | 100% (44% to 100%) | 93% |
| prompt+alexandria_profile:notes+task+restyle+examples | 3 | 67% (21% to 94%) | 0% (0% to 56%) | 20% |
| prompt+alexandria_profile:finalized+task+restyle+plain-words | 3 | 33% (6% to 79%) | 33% (6% to 79%) | 55% |
| prompt+alexandria_profile:notes+task+restyle+plain-words | 3 | 67% (21% to 94%) | 0% (0% to 56%) | 30% |
| full_document+alexandria_profile:finalized+writer+t1.3 | 3 | 0% (0% to 56%) | 33% (6% to 79%) | 71% |
| full_document+alexandria_profile:finalized+writer+t1.3+revise3-sentences | 3 | 100% (44% to 100%) | 0% (0% to 56%) | 10% |
| filled_answers+alexandria_profile | 2 | 100% (34% to 100%) | 0% (0% to 66%) | 11% |
| full_document+alexandria_profile+roundtrip-german+clauses+rules+math+revise10 | 2 | 100% (34% to 100%) | 0% (0% to 66%) | 9% |
| full_document+alexandria_profile+clauses+rules+math+revise10 | 2 | 50% (9% to 91%) | 0% (0% to 66%) | 27% |
| prompt+pasted_style_profile | 1 | 0% (0% to 79%) | 100% (21% to 100%) | 98% |
| prompt+alexandria_profile+sub_5_percent_stealth_v3 | 1 | 100% (21% to 100%) | 0% (0% to 79%) | 2% |
| prompt+alexandria_profile+math+revise10 | 1 | 0% (0% to 79%) | 100% (21% to 100%) | 96% |
| full_document+alexandria_profile:finalized+writer+t0.8 | 1 | 0% (0% to 79%) | 100% (21% to 100%) | 98% |
| full_document+alexandria_profile:finalized+writer+t0.8+revise3-sentences | 1 | 100% (21% to 100%) | 0% (0% to 79%) | 16% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page | 1 | 0% (0% to 79%) | 100% (21% to 100%) | 100% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+revise10-sentences | 1 | 100% (21% to 100%) | 0% (0% to 79%) | 8% |

### By model

| Model | Samples | Missed | Caught | Mean AI likelihood |
|---|---:|---:|---:|---:|
| gemini-3.5-flash-lite | 144 | 35% (27% to 43%) | 44% (36% to 52%) | 59% |
| gemini-3.1-flash-lite | 92 | 45% (35% to 55%) | 38% (29% to 48%) | 51% |
| openai/gpt-oss-120b | 3 | 0% (0% to 56%) | 100% (44% to 100%) | 97% |
| openai/gpt-oss-20b | 3 | 0% (0% to 56%) | 100% (44% to 100%) | 99% |
| mixed | 2 | 100% (34% to 100%) | 0% (0% to 66%) | 14% |
| gemini-web | 2 | 0% (0% to 66%) | 100% (34% to 100%) | 98% |
| gemini-3-flash-preview | 1 | 100% (21% to 100%) | 0% (0% to 79%) | 2% |

### By thinking

| Thinking | Samples | Missed | Caught | Mean AI likelihood |
|---|---:|---:|---:|---:|
| low | 235 | 37% (32% to 44%) | 43% (37% to 49%) | 57% |
| minimal | 7 | 43% (16% to 75%) | 43% (16% to 75%) | 53% |
| not recorded | 4 | 75% (30% to 95%) | 25% (5% to 70%) | 31% |
| medium | 1 | 0% (0% to 79%) | 100% (21% to 100%) | 85% |

### By temperature

| Temperature | Samples | Missed | Caught | Mean AI likelihood |
|---|---:|---:|---:|---:|
| 0.8 | 227 | 36% (30% to 42%) | 44% (38% to 51%) | 59% |
| 1.0 | 10 | 70% (40% to 89%) | 30% (11% to 60%) | 33% |
| 1.3 | 6 | 50% (19% to 81%) | 17% (3% to 56%) | 41% |
| not recorded | 4 | 75% (30% to 95%) | 25% (5% to 70%) | 31% |

### By blocks per request

| Blocks per request | Samples | Missed | Caught | Mean AI likelihood |
|---|---:|---:|---:|---:|
| not recorded | 243 | 37% (32% to 44%) | 44% (38% to 50%) | 57% |
| 8 | 4 | 75% (30% to 95%) | 0% (0% to 49%) | 18% |

### By dataset

| Dataset | Samples | Missed | Caught | Mean AI likelihood |
|---|---:|---:|---:|---:|
| local_tuning | 235 | 37% (32% to 44%) | 43% (37% to 49%) | 57% |
| local_redteam | 12 | 50% (25% to 75%) | 42% (19% to 68%) | 48% |

## Human originals

- Wrongly flagged (Likely AI): 0% (0% to 56%)
- Needs review: 0% (0% to 56%)
- Mean AI likelihood: 3%

