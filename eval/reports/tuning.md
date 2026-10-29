# Generation settings

Measured 2026-10-29 on 16 blocks (447 words) of `AI_REDTEAM_full_document_20261029T150118Z_2985a495f436ed2e.docx`, with the `alexandria_profile` style profile and the slim detector. Written by `python -m eval.tune_generation`.

**Default (points spread evenly):** gemini-3.1-flash-lite, minimal thinking, 8 blocks per request, 2 requests at once.

All points on one measure:

- detector: gemini-3.1-flash-lite, high thinking, 8 blocks per request
- style: gemini-3.1-flash-lite, low thinking, 4 blocks per request
- speed: gemini-3.1-flash-lite, minimal thinking, 8 blocks per request
- quota: gemini-3.1-flash-lite, minimal thinking, 8 blocks per request

AI likelihood is the detector's, lower is better. Style match is the share of the writer's own passages that are further from the profile, higher is better. ± is the spread between repeats: differences smaller than it are noise. Predicted rows add the group size's effect measured on gemini-3.5-flash-lite.

Free requests a day, from Gemini's quota errors or set by hand: gemini-3-flash-preview 20, gemini-3.6-flash 20. A model whose limit isn't known loses no quota score for it. The picks and scores here are for a 100-block batch; the Red team page scores the document it has.

| Model | Thinking | Blocks per request | AI likelihood % | Style match % | Seconds per block | Requests per block | Tokens per block | Trials | Even-points score |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gemini-3.1-flash-lite | minimal | 8 | 5 ± 2 | 39 ± 8 | 0.7 | 0.12 | 2,046 | 2 | 0.83 |
| gemini-3.6-flash | minimal | 8 | 3 ± 0 | 46 ± 16 | 0.8 | 0.12 | 2,056 | 2 | 0.77 |
| gemini-3.1-flash-lite | medium | 8 | 4 ± 3 | 29 ± 25 | 1.1 | 0.12 | 2,242 | 2 | 0.72 |
| gemini-3.1-flash-lite | minimal | 4 | 6 | 48 | 1.0 | 0.25 | 3,977 | predicted | 0.70 |
| gemini-3-flash-preview | minimal | 8 | 3 | 35 | 1.2 | 0.12 | 2,046 | 1 | 0.67 |
| gemini-3.5-flash-lite | minimal | 8 | 3 ± 0 | 40 ± 6 | 2.4 | 0.12 | 2,054 | 2 | 0.67 |
| gemini-3.5-flash-lite | medium | 8 | 3 ± 0 | 47 ± 0 | 3.4 | 0.12 | 2,277 | 2 | 0.66 |
| gemini-3.1-flash-lite | low | 8 | 4 ± 2 | 48 ± 5 | 4.3 | 0.12 | 2,319 | 2 | 0.64 |
| gemini-3.1-flash-lite | high | 8 | 2 ± 0 | 36 ± 5 | 2.6 | 0.12 | 2,820 | 2 | 0.63 |
| gemini-3.1-flash-lite | medium | 4 | 5 | 38 | 1.6 | 0.25 | 4,357 | predicted | 0.61 |
| gemini-3.5-flash-lite | low | 8 | 3 ± 0 | 25 ± 28 | 3.2 | 0.12 | 2,057 | 2 | 0.61 |
| gemini-3.6-flash | minimal | 4 | 4 | 55 | 1.2 | 0.25 | 3,996 | predicted | 0.61 |
| gemini-3.5-flash-lite | high | 8 | 4 ± 1 | 30 ± 6 | 3.8 | 0.12 | 2,427 | 2 | 0.60 |
| gemini-3.5-flash-lite | minimal | 4 | 4 | 49 | 3.5 | 0.25 | 3,992 | predicted | 0.58 |
| gemini-3.5-flash-lite | medium | 4 | 4 | 56 | 4.9 | 0.25 | 4,425 | predicted | 0.58 |
| gemini-3.1-flash-lite | low | 4 | 5 | 57 | 6.3 | 0.25 | 4,507 | predicted | 0.57 |
| gemini-3.1-flash-lite | high | 4 | 3 | 46 | 3.8 | 0.25 | 5,481 | predicted | 0.56 |
| gemini-3-flash-preview | minimal | 4 | 4 | 44 | 1.8 | 0.25 | 3,977 | predicted | 0.54 |
| gemini-3.5-flash-lite | low | 4 | 4 ± 2 | 34 ± 15 | 4.6 | 0.25 | 3,998 | 2 | 0.53 |
| gemini-3.5-flash-lite | high | 4 | 5 | 39 | 5.5 | 0.25 | 4,716 | predicted | 0.53 |
| gemini-3.5-flash-lite | medium | 1 | 3 | 39 | 27.6 | 1.00 | 18,080 | predicted | 0.45 |
| gemini-3.1-flash-lite | minimal | 1 | 6 | 31 | 5.9 | 1.00 | 16,248 | predicted | 0.45 |
| gemini-3.1-flash-lite | low | 1 | 5 | 40 | 35.4 | 1.00 | 18,411 | predicted | 0.45 |
| gemini-3.5-flash-lite | minimal | 1 | 3 | 32 | 19.8 | 1.00 | 16,309 | predicted | 0.43 |
| gemini-3.1-flash-lite | high | 1 | 2 | 28 | 21.4 | 1.00 | 22,391 | predicted | 0.42 |
| gemini-3.1-flash-lite | medium | 1 | 4 | 21 | 8.9 | 1.00 | 17,801 | predicted | 0.42 |
| gemini-3.5-flash-lite | high | 1 | 4 | 22 | 31.1 | 1.00 | 19,266 | predicted | 0.40 |
| gemini-3.5-flash-lite | low | 1 | 3 ± 1 | 17 ± 7 | 26.2 | 1.00 | 16,333 | 2 | 0.39 |
| gemini-3.6-flash | minimal | 1 | 3 | 38 | 6.8 | 1.00 | 16,327 | predicted | 0.38 |
| gemini-3-flash-preview | minimal | 1 | 4 | 26 | 9.9 | 1.00 | 16,248 | predicted | 0.35 |
