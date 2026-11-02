# Revising rewrites against the detector

20 runs on 7 AI-written documents, revising by sentences. The bar is an AI likelihood under 10%. Written by `python -m eval.tune_revise`. Brackets are Wilson 95% intervals, which are too narrow in as far as runs of one document resemble each other.

| Document | Runs | Written % | Revised % | Range % | Under the bar | Full detector % | Style distance before > after | Names, numbers kept % | Length |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| claude_salamis.md | 3 | 86 | 9 | 5-11 | 2 | 7 | 1.32 > 1.23 | 76 | x0.98 |
| mage-10520.md | 3 | 73 | 36 | 22-63 | 0 | 29 | 1.77 > 1.59 | 66 | x0.93 |
| mage-13863.md | 3 | 8 | 7 | 3-10 | 3 | 5 | 1.70 > 1.60 | 67 | x0.99 |
| mage-15379.md | 3 | 100 | 60 | 30-94 | 0 | 53 | 1.46 > 1.32 | 35 | x0.96 |
| mage-19186.md | 3 | 100 | 58 | 5-91 | 1 | 63 | 4.22 > 2.23 | - | x0.60 |
| mage-2169.md | 3 | 50 | 5 | 4-6 | 3 | 4 | 1.30 > 0.87 | 56 | x0.98 |
| mage-22431.md | 2 | 100 | 64 | 44-85 | 0 | 61 | 2.27 > 2.08 | 21 | x0.64 |
| **All** | 20 | 72 | 33 | 3-94 | 9 | 30 | 1.99 > 1.53 | 55 | x0.88 |

- **Under the bar:** 9 of 20 runs (26% to 66%).
- **Documents whose mean is under the bar:** 3 of 7.
- **Median run:** 16.2%.
