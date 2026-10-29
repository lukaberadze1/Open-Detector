# With and without the style profile

Measured 2026-10-29 with `python -m eval.tune_generation`, once as usual and once with `--no-style`, on
the same 16 prose blocks (466 words) of a document: low thinking, temperature 0.8, 8 blocks per request,
3 repeats each, scored with the slim detector. Both runs are measured against the `alexandria_profile`
style profile; the second just doesn't send it.

| Model | Style profile | AI likelihood % | Style match % | Seconds per block | Tokens per block |
|---|---|---:|---:|---:|---:|
| gemini-3.5-flash-lite | sent | 3 ± 0 | 36 ± 0 | 0.5 | 1,950 |
| gemini-3.5-flash-lite | not sent | 96 ± 2 | 0 ± 0 | 0.4 | 821 |
| gemini-3.1-flash-lite | sent | 2 ± 0 | 31 ± 12 | 1.0 | 2,192 |
| gemini-3.1-flash-lite | not sent | 98 ± 2 | 0 ± 0 | 1.0 | 991 |

± is the standard deviation between the three repeats.

- **The style profile is what gets the text past the detector.** Without it the detector catches every
  rewrite (96 to 98% AI likelihood); with it, 2 to 3%. Temperature (0.8 to 1.3) and thinking level moved
  these numbers by a few points at most.
- **Without the profile nothing reads like the writer**: every rewrite is further from the profile than
  any of the writer's own passages (0%). With it, a third of the writer's passages are further away.
- **It costs about 1,100 tokens a block** at 8 blocks per request (the profile is sent once per request),
  and no extra requests or time.

For the detector, this is the gap to close: plain AI rewrites are caught, rewrites in a measured human
style are not. Every rewrite from both runs is saved as labelled AI data in
`data/essays/tuning_redteam.jsonl` (attack `full_document+alexandria_profile` and
`full_document+plain`), and `python -m eval.redteam_report` tracks the miss rate as the detector is
retrained.

## The old profile format against the two registers

Measured 2026-10-30 on the same 16 blocks, after the profile took in the writer's Obsidian notes up to
2026-W44, with the same settings and 3 repeats each. The old format sends everything; the new one
(`style_profile_template.json`) sends one register and what holds in both.

| Model | Profile | AI likelihood % | Style match % | Tokens per block |
|---|---|---:|---:|---:|
| gemini-3.5-flash-lite | old format | 3 ± 1 | 40 ± 7 | 2,040 |
| gemini-3.5-flash-lite | new, finished work | 2 ± 0 | 46 ± 2 | 1,583 |
| gemini-3.5-flash-lite | new, quick notes | 2 ± 0 | 45 ± 6 | 1,526 |
| gemini-3.1-flash-lite | old format | 2 ± 0 | 46 ± 7 | 2,271 |
| gemini-3.1-flash-lite | new, finished work | 3 ± 1 | 40 ± 9 | 1,736 |
| gemini-3.1-flash-lite | new, quick notes | 1 ± 0 | 31 ± 15 | 1,708 |

- **Same strength, 22 to 25% fewer tokens a block.** The detector misses every version (1 to 3%), and
  the style match differences are inside the spread between repeats.
- The style match is measured against the writer's notes, the only raw writing the profile has, so it
  can't yet say how well the finished-work register matches finished work. Finished pieces saved in
  `data/styles/alexandria/finalized/` would let it.
