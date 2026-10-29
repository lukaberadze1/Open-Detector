# Red-team samples against the detector

13 AI samples and 0 human ones of 150 words or more, scored with the slim detector and its shipped thresholds. Written by `python -m eval.redteam_report`. Brackets are Wilson 95% intervals.

## AI samples

- Caught (Likely AI): 23% (8% to 50%)
- Needs review: 8% (1% to 33%)
- **Missed (Likely human): 69% (42% to 87%)**
- Mean AI likelihood: 32%

### By attack

| Attack | Samples | Missed | Caught | Mean AI likelihood |
|---|---:|---:|---:|---:|
| full_markdown+pasted_style_profile | 5 | 60% (23% to 88%) | 40% (12% to 77%) | 44% |
| regenerate_section+pasted_style_profile | 3 | 67% (21% to 94%) | 0% (0% to 56%) | 24% |
| filled_answers+alexandria_profile | 3 | 100% (44% to 100%) | 0% (0% to 56%) | 9% |
| prompt+pasted_style_profile | 1 | 0% (0% to 79%) | 100% (21% to 100%) | 98% |
| prompt+alexandria_profile+sub_5_percent_stealth_v3 | 1 | 100% (21% to 100%) | 0% (0% to 79%) | 2% |

### By model

| Model | Samples | Missed | Caught | Mean AI likelihood |
|---|---:|---:|---:|---:|
| gemini-3.5-flash-lite | 7 | 71% (36% to 92%) | 14% (3% to 51%) | 26% |
| gemma3:4b | 3 | 33% (6% to 79%) | 67% (21% to 94%) | 71% |
| gemini-3.6-flash | 1 | 100% (21% to 100%) | 0% (0% to 79%) | 1% |
| gemini-3-flash-preview | 1 | 100% (21% to 100%) | 0% (0% to 79%) | 2% |
| mixed | 1 | 100% (21% to 100%) | 0% (0% to 79%) | 21% |

