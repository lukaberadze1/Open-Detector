# humanize-text's demo data against the slim detector

Likely AI is above 0.796, Likely human at or under 0.150. Written by `python -m eval.score_demo`. Only 50 inputs, so a rough reading. Rewrites by their model count as AI.

| Texts | n | Mean | Median | Likely AI | Likely human | Under 0.5 |
|---|---:|---:|---:|---:|---:|---:|
| AI source | 44 | 0.623 | 0.821 | 23 | 10 | 16 |
| Human rewrite | 44 | 0.268 | 0.074 | 8 | 30 | 33 |
| Their model, temperature 0.3 | 44 | 0.468 | 0.496 | 13 | 16 | 22 |
| Their model, temperature 0.5 | 44 | 0.474 | 0.456 | 11 | 13 | 25 |
| Their model, temperature 0.7 | 44 | 0.372 | 0.283 | 9 | 16 | 31 |
| Their model, temperature 0.9 | 45 | 0.354 | 0.203 | 9 | 20 | 32 |
| Their model, temperature 1.0 | 89 | 0.344 | 0.178 | 16 | 42 | 61 |
