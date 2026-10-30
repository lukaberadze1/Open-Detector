# With and without the style profile

## Correction, 2026-10-30: the profiled rewrites were copies

The two sections below say the style profile took the detector from 96-98% to 2-3%. It didn't. Told to
write like the writer of the document it was rewriting, the model handed the document back: every
profiled trial logged (42 of 42) keeps 90% or more of the original's words, and gemini-3.5-flash-lite
returned 24 to 31 of 32 paragraphs word for word. The detector was scoring the writer's own report, and
rightly called it human. The style match of about 45% is the report's own: it scores 45.5% unchanged.

Measured again on the same 16 blocks (466 words) of the writer's report, low thinking, temperature 0.8,
8 blocks per request, 3 repeats per model, before the copy guard below existed. *Rewrite* is how closely
each rewrite follows the original (eval/redteam_batch.py `STRUCTURES`); *words kept* is the share of
the original's words a rewrite still has.

| Profile | Rewrite | AI likelihood % (3.5 / 3.1 Flash-Lite) | Style match % | Structure match % | Words kept % | Copies |
|---|---|---:|---:|---:|---:|---:|
| none | free | 95 ± 3 / 100 ± 0 | 0 / 0 | 80 / 85 | 58 / 56 | 0 of 6 |
| none | same structure | 3 ± 1 / 8 ± 8 | 0 / 0 | 99 / 99 | 48 / 45 | 0 of 6 |
| none | light edit | 3 ± 1 / 6 ± 2 | 0 / 0 | 100 / 98 | 93 / 68 | 3 of 6 |
| notes register | free | 2 ± 1 / 2 ± 1 | 32 / 33 | 95 / 89 | 100 / 97 | 6 of 6 |
| notes register | same structure | 2 ± 1 / 2 ± 0 | 3 / 23 | 99 / 99 | 52 / 76 | 1 of 6 |

- **What gets AI text past the detector is the original's sentence structure, not the profile.** With
  no profile at all, writing each of the original's sentences anew, one for one at its length, takes
  the detector from 95-100% to 3-8%, while keeping fewer of the original's words (45-48%) than a free
  rewrite does (56-58%). The profile on top of that adds little: 2% against 3-8%.
- **A free rewrite in the writer's own style is a copy.** That is not AI text, and saved as AI data it
  would teach the detector that the writer's words are AI. Since this measurement a rewrite that keeps
  90% or more of its original (`COPIED`) is written again once, in new words, and if it still comes
  back as the original it is never saved as AI; `eval.datasets.load_local` and `eval.redteam_report`
  leave such rows out (82 of the Red team page's 210 saved AI rows, and the 42 trials above).
- **The style match measures against the writer's notes**, so a rewrite that follows a finished report
  scores low however human it reads. Since 2026-10-30 a finished-work rewrite is measured against the
  finished pieces in `data/styles/alexandria/finalized/` instead.

Since then a free rewrite is also told to use new words (`FOLLOW["free"]`), and copies from one request
are asked for again together (`rewrite_group`). Measured again on the writer's report, notes register,
free rewrite:

| Model | AI likelihood % | Style match % | Structure match % | Words kept % | Copies | Tokens per block |
|---|---:|---:|---:|---:|---:|---:|
| gemini-3.5-flash-lite | 6 ± 4 | 0 | 87 | 41 | 0 of 3 | 1,524 |
| gemini-3.1-flash-lite | 3 ± 2 | 14 | 92 | 69 | 0 of 3 | 3,303 |

Real rewrites now, about half of the original's words changed, and still missed by the detector.

## An AI-written source, 2026-10-30

A 736-word report on the Battle of Salamis, written by Claude and scored 100% AI by the slim detector
(`data/essays/sources/claude_salamis.md`), rewritten with the same settings, 16 blocks (about 650 words),
3 repeats per model. The first runs asked again for each copy on its own; the last two rows are after
the grouped retry.

| Profile | Rewrite | AI likelihood % (3.5 / 3.1 Flash-Lite) | Structure match % | Words kept % | Tokens per block |
|---|---|---:|---:|---:|---:|
| none | free | 99 ± 1 / 100 ± 0 | 83 / 83 | 42 / 50 | 527 / 524 |
| none | same structure | 87 ± 10 / 98 ± 1 | 93 / 97 | 33 / 45 | 587 / 587 |
| notes register | free | 73 ± 17 / 42 ± 7 | 78 / 63 | 71 / 77 | 6,547 / 6,338 |
| notes register | same structure | 63 ± 24 / 60 ± 19 | 95 / 94 | 39 / 72 | 2,024 / 6,124 |
| finished-work register | free | 40 ± 24 / 91 ± 7 | 67 / 52 | 65 / 69 | 5,280 / 2,852 |
| finished-work register | same structure | 53 ± 34 / 35 ± 21 | 94 / 92 | 42 / 69 | 2,388 / 7,612 |
| notes register | free, grouped retry | 84 ± 19 / 61 ± 14 | 79 / 66 | 37 / 62 | 1,992 / 2,678 |
| finished-work register | same structure, grouped retry | 64 ± 22 / 46 ± 38 | 94 / 91 | 42 / 64 | 2,360 / 3,025 |

- **On an AI source the profile does the work.** Without it every rewrite is caught (87-100%); with
  either register the AI likelihood falls to 35-84% on average, mostly into *Needs review* (33-80% on
  the slim build), with a wide spread between repeats: some rewrites pass, some don't.
- **The original's structure matters more than the profile.** Keeping a human original's sentences
  took a rewrite to 3-8% with no profile at all; keeping an AI original's sentences leaves it at 87-98%.
  The detector reads the skeleton as much as the words.
- **Asking for each copy on its own cost 4 to 12 times the tokens**, because each request carries the
  profile and the document. Retrying the copies together costs about twice a plain run.

The sections below are kept as they were measured.

## The first measurement

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
