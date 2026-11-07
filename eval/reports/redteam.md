# Red-team samples against the detector

657 AI samples and 3 human ones of 150 words or more, scored with the slim detector and its shipped thresholds. Written by `python -m eval.redteam_report`. Brackets are Wilson 95% intervals.

## AI samples

- Caught (Likely AI): 51% (48% to 55%)
- Needs review: 26% (23% to 29%)
- **Missed (Likely human): 23% (20% to 26%)**
- Mean AI likelihood: 63%

### By attack

| Attack | Samples | Missed | Caught | Mean AI likelihood |
|---|---:|---:|---:|---:|
| humanizer+academic-control | 40 | 0% (0% to 9%) | 82% (68% to 91%) | 89% |
| humanizer+stealth-rules | 40 | 5% (1% to 17%) | 88% (74% to 95%) | 90% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+best3+drafts3 | 39 | 13% (6% to 27%) | 69% (54% to 81%) | 80% |
| prompt+human_default_profile:finalized+task+restyle | 26 | 15% (6% to 34%) | 19% (9% to 38%) | 45% |
| prompt+alexandria_profile:notes+task+restyle | 26 | 35% (19% to 54%) | 12% (4% to 29%) | 31% |
| prompt+alexandria_profile:finalized+task+restyle | 25 | 12% (4% to 30%) | 36% (20% to 55%) | 57% |
| full_document+alexandria_profile:finalized+writer+t0.8+as-page | 20 | 15% (5% to 36%) | 65% (43% to 82%) | 72% |
| full_document+alexandria_profile:finalized+writer+t0.8+as-page+revise4-sentences | 20 | 50% (30% to 70%) | 15% (5% to 36%) | 33% |
| full_document+plain | 18 | 0% (0% to 18%) | 100% (82% to 100%) | 98% |
| full_document+alexandria_profile:notes | 18 | 33% (16% to 56%) | 17% (6% to 39%) | 45% |
| full_document+alexandria_profile:finalized+writer | 18 | 44% (25% to 66%) | 22% (9% to 45%) | 35% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+best3 | 15 | 0% (0% to 20%) | 87% (62% to 96%) | 87% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+best3+revise10-sentences | 15 | 80% (55% to 93%) | 7% (1% to 30%) | 14% |
| humanizer+cive-bart+sampled | 15 | 0% (0% to 20%) | 93% (70% to 99%) | 95% |
| humanizer+cive-bart+best8 | 15 | 0% (0% to 20%) | 93% (70% to 99%) | 97% |
| humanizer+stealth-bart+best8 | 14 | 0% (0% to 22%) | 100% (78% to 100%) | 99% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+best3+drafts3+revise10-sentences | 13 | 85% (58% to 96%) | 0% (0% to 23%) | 13% |
| humanizer+stealth-bart+sampled | 13 | 0% (0% to 23%) | 92% (67% to 99%) | 97% |
| full_document+plain+same | 12 | 42% (19% to 68%) | 42% (19% to 68%) | 49% |
| full_document+alexandria_profile:finalized+same | 12 | 17% (5% to 45%) | 8% (1% to 35%) | 50% |
| prompt+human_default_profile:notes+task+restyle | 12 | 17% (5% to 45%) | 50% (25% to 75%) | 65% |
| full_document+alexandria_profile:notes+same | 11 | 45% (21% to 72%) | 18% (5% to 48%) | 34% |
| prompt+human_default_profile:finalized+task+restyle+persona | 10 | 10% (2% to 40%) | 70% (40% to 89%) | 83% |
| prompt+alexandria_profile:finalized+challenge+old-voice | 9 | 0% (0% to 30%) | 78% (45% to 94%) | 86% |
| prompt+alexandria_profile:finalized+task | 9 | 0% (0% to 30%) | 89% (56% to 98%) | 91% |
| prompt+alexandria_profile:finalized+task+plan | 9 | 0% (0% to 30%) | 89% (56% to 98%) | 92% |
| prompt+human_default_profile:finalized+challenge+old-voice | 9 | 0% (0% to 30%) | 56% (27% to 81%) | 81% |
| prompt+alexandria_profile:notes+task | 9 | 11% (2% to 44%) | 22% (6% to 55%) | 61% |
| prompt+alexandria_profile:finalized+task+restyle+revise3 | 9 | 67% (35% to 88%) | 0% (0% to 30%) | 21% |
| prompt+alexandria_profile:notes+task+restyle+revise3 | 9 | 67% (35% to 88%) | 0% (0% to 30%) | 18% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+best3+roundtrip-german | 9 | 11% (2% to 44%) | 78% (45% to 94%) | 78% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+best3+roundtrip-german+revise10-sentences | 9 | 89% (56% to 98%) | 11% (2% to 44%) | 17% |
| prompt+human_default_profile:notes+task | 8 | 0% (0% to 32%) | 88% (53% to 98%) | 92% |
| full_document+alexandria_profile:finalized | 6 | 0% (0% to 39%) | 50% (19% to 81%) | 65% |
| full_document+plain+writer | 6 | 0% (0% to 39%) | 100% (61% to 100%) | 94% |
| full_document+alexandria_profile:notes+writer | 6 | 0% (0% to 39%) | 33% (10% to 70%) | 56% |
| full_document+human_default_profile:finalized+writer | 6 | 0% (0% to 39%) | 67% (30% to 90%) | 76% |
| full_document+human_default_profile:finalized+same | 6 | 0% (0% to 39%) | 100% (61% to 100%) | 92% |
| full_document+human_default_profile:notes+writer | 6 | 0% (0% to 39%) | 50% (19% to 81%) | 72% |
| prompt+alexandria_profile:finalized+task+restyle+plain-words | 6 | 33% (10% to 70%) | 17% (3% to 56%) | 45% |
| prompt+alexandria_profile:notes+task+restyle+plain-words | 6 | 67% (30% to 90%) | 0% (0% to 39%) | 19% |
| full_document+alexandria_profile:finalized+writer+revise3-sentences | 6 | 83% (44% to 97%) | 0% (0% to 39%) | 7% |
| full_document+alexandria_profile:finalized+writer+revise3-windows | 6 | 83% (44% to 97%) | 0% (0% to 39%) | 7% |
| prompt+human_default_profile:finalized+task | 6 | 0% (0% to 39%) | 100% (61% to 100%) | 98% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+masked+best3 | 5 | 20% (4% to 62%) | 40% (12% to 77%) | 59% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+masked+best3+revise10-sentences | 5 | 100% (57% to 100%) | 0% (0% to 43%) | 6% |
| full_document+alexandria_profile:finalized+writer+t1 | 4 | 0% (0% to 49%) | 75% (30% to 95%) | 70% |
| full_document+alexandria_profile:finalized+writer+t1+revise3-sentences | 4 | 100% (51% to 100%) | 0% (0% to 49%) | 7% |
| prompt+alexandria_profile | 3 | 0% (0% to 56%) | 100% (44% to 100%) | 95% |
| full_document+plain+light | 3 | 100% (44% to 100%) | 0% (0% to 56%) | 6% |
| full_document+human_default_profile:finalized+writer+persona | 3 | 0% (0% to 56%) | 100% (44% to 100%) | 94% |
| prompt+alexandria_profile:finalized+task+restyle+examples | 3 | 0% (0% to 56%) | 100% (44% to 100%) | 93% |
| prompt+alexandria_profile:notes+task+restyle+examples | 3 | 67% (21% to 94%) | 0% (0% to 56%) | 20% |
| full_document+alexandria_profile:finalized+writer+t1.3 | 3 | 0% (0% to 56%) | 33% (6% to 79%) | 71% |
| full_document+alexandria_profile:finalized+writer+t1.3+revise3-sentences | 3 | 100% (44% to 100%) | 0% (0% to 56%) | 10% |
| filled_answers+alexandria_profile | 2 | 50% (9% to 91%) | 0% (0% to 66%) | 11% |
| full_document+alexandria_profile+roundtrip-german+clauses+rules+math+revise10 | 2 | 100% (34% to 100%) | 0% (0% to 66%) | 9% |
| full_document+alexandria_profile+clauses+rules+math+revise10 | 2 | 50% (9% to 91%) | 0% (0% to 66%) | 27% |
| full_document+alexandria_profile+clauses+rules+math+revise10+masked | 2 | 0% (0% to 66%) | 50% (9% to 91%) | 68% |
| prompt+pasted_style_profile | 1 | 0% (0% to 79%) | 100% (21% to 100%) | 98% |
| prompt+alexandria_profile+sub_5_percent_stealth_v3 | 1 | 100% (21% to 100%) | 0% (0% to 79%) | 2% |
| prompt+alexandria_profile+math+revise10 | 1 | 0% (0% to 79%) | 100% (21% to 100%) | 96% |
| full_document+human_default_profile:notes+writer+persona | 1 | 0% (0% to 79%) | 0% (0% to 79%) | 76% |
| full_document+alexandria_profile:finalized+writer+t0.8 | 1 | 0% (0% to 79%) | 100% (21% to 100%) | 98% |
| full_document+alexandria_profile:finalized+writer+t0.8+revise3-sentences | 1 | 0% (0% to 79%) | 0% (0% to 79%) | 16% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page | 1 | 0% (0% to 79%) | 100% (21% to 100%) | 100% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+revise10-sentences | 1 | 100% (21% to 100%) | 0% (0% to 79%) | 8% |

### By model

| Model | Samples | Missed | Caught | Mean AI likelihood |
|---|---:|---:|---:|---:|
| gemini-3.5-flash-lite | 301 | 25% (20% to 30%) | 43% (37% to 49%) | 57% |
| gemini-3.1-flash-lite | 208 | 34% (28% to 41%) | 38% (32% to 45%) | 51% |
| AI-Text-Humanizer-App | 40 | 0% (0% to 9%) | 82% (68% to 91%) | 89% |
| StealthHumanizer-local | 40 | 5% (1% to 17%) | 88% (74% to 95%) | 90% |
| cive202/humanize-ai-text-bart-large | 30 | 0% (0% to 11%) | 93% (79% to 98%) | 96% |
| rudra496/stealthhumanizer-bart | 27 | 0% (0% to 12%) | 96% (82% to 99%) | 98% |
| openai/gpt-oss-120b | 3 | 0% (0% to 56%) | 100% (44% to 100%) | 97% |
| openai/gpt-oss-20b | 3 | 0% (0% to 56%) | 100% (44% to 100%) | 99% |
| mixed | 2 | 50% (9% to 91%) | 0% (0% to 66%) | 14% |
| gemini-web | 2 | 0% (0% to 66%) | 100% (34% to 100%) | 98% |
| gemini-3-flash-preview | 1 | 100% (21% to 100%) | 0% (0% to 79%) | 2% |

### By thinking

| Thinking | Samples | Missed | Caught | Mean AI likelihood |
|---|---:|---:|---:|---:|
| low | 506 | 28% (25% to 32%) | 42% (37% to 46%) | 55% |
| not recorded | 141 | 3% (1% to 7%) | 87% (81% to 92%) | 91% |
| minimal | 9 | 33% (12% to 65%) | 44% (19% to 73%) | 56% |
| medium | 1 | 0% (0% to 79%) | 100% (21% to 100%) | 85% |

### By temperature

| Temperature | Samples | Missed | Caught | Mean AI likelihood |
|---|---:|---:|---:|---:|
| 0.8 | 500 | 27% (24% to 31%) | 42% (38% to 47%) | 56% |
| not recorded | 141 | 3% (1% to 7%) | 87% (81% to 92%) | 91% |
| 1.0 | 10 | 60% (31% to 83%) | 30% (11% to 60%) | 33% |
| 1.3 | 6 | 50% (19% to 81%) | 17% (3% to 56%) | 41% |

### By blocks per request

| Blocks per request | Samples | Missed | Caught | Mean AI likelihood |
|---|---:|---:|---:|---:|
| not recorded | 651 | 23% (20% to 26%) | 52% (48% to 56%) | 63% |
| 8 | 6 | 50% (19% to 81%) | 17% (3% to 56%) | 35% |

### By dataset

| Dataset | Samples | Missed | Caught | Mean AI likelihood |
|---|---:|---:|---:|---:|
| local_tuning | 506 | 28% (25% to 32%) | 42% (37% to 46%) | 55% |
| humanizer_redteam | 137 | 1% (0% to 5%) | 89% (83% to 93%) | 93% |
| local_redteam | 14 | 36% (16% to 61%) | 43% (21% to 67%) | 51% |

## Human originals

- Wrongly flagged (Likely AI): 0% (0% to 56%)
- Needs review: 0% (0% to 56%)
- Mean AI likelihood: 3%

