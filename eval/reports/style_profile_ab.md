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

## Both ways of writing, both profiles, 2026-10-30

Standard generation (a new piece from instructions) was measured for the first time with
`python -m eval.tune_prompts`: 3 prompts of each kind (an essay, a report, a story; a review, a forum
reply, an explanation) × 3 repeats on gemini-3.5-flash-lite, each sample scored on its own by the slim
detector, which calls 33-80% *Needs review* and under 33% *Likely human*. "Built-in" is the human
profile measured from 1,750 writers (`styles/human_default_profile.json`).

| Profile, register | Prompts | Before: AI likelihood % | Likely human | After restyle: AI likelihood % | Likely human | Words | Tokens per sample |
|---|---|---:|---:|---:|---:|---:|---:|
| yours, finished work | essay, report, story | 91 ± 12 | 0 of 9 | 42 ± 26 | 4 of 9 | 331 | 23,059 |
| yours, quick notes | review, reply, explanation | 61 ± 28 | 2 of 9 | 40 ± 33 | 5 of 9 | 316 | 21,327 |
| built-in, finished work | essay, report, story | 81 ± 18 | 0 of 9 | 43 ± 34 | 4 of 9 | 229 | 12,472 |
| built-in, quick notes | review, reply, explanation | 92 ± 14 | 0 of 8 | 68 ± 36 | 2 of 9 | 226 | 10,641 |

Document rewrites, 16 blocks, 3 repeats on each of 3.5 and 3.1 Flash-Lite:

| Original | Profile, register | Same structure | Writer's sentences |
|---|---|---:|---:|
| the writer's own report | yours, finished work | 3-8 (no profile) | 1 ± 1 / 9 ± 6 |
| Claude's report (100% AI) | yours, finished work | 53 / 35 | 16 ± 13 / 22 ± 9 |
| Claude's report | yours, quick notes | 63 / 60 | 63 / 49 |
| Claude's report | built-in, finished work | 94 / 91 | 68 / 85 |
| Claude's report | built-in, quick notes | | 90 / 53 |

Why they were weak:

- **Written from instructions, a sample is the model's own plan of what to say**: complete, balanced,
  each point in order. The detector reads that plan whatever voice the model is told to write it in.
  Telling the model more about the voice changed nothing: naming it an essay rather than a "challenge
  sample", or giving it a whole-sample sentence plan, left it at 86-92%.
- **The page told a sample from instructions not to say "I" or "you"**: its voice rule, written for
  rewriting a source, said to add no first or second person "if it is absent" from a source there
  wasn't. People write about 19 "I" and 16 "you" per 1,000 words in informal writing.
- **A rewrite in place keeps an AI original's skeleton**: two even 20-word sentences per block survive
  any rewording. Borrowing the writer's sentence shapes breaks that, but only with the writer's voice
  too: the shapes alone left it at 91-97%.
- **The built-in profile is everyone's average, and the average of many writers is no one**: clean,
  neutral, standard prose with the quirks averaged away, which is how the model writes anyway. A
  writer's own profile is a list of quirks (98% "a" before vowels, no apostrophes in notes, "However"
  with no comma, spaces before full stops), and that is what it has over the built-in.

What changed:

- A sample from instructions is drafted, then rewritten passage by passage in the writer's voice and
  borrowed sentence shapes (`restyle`), as the document rewrites that pass are. The draft is asked for
  a quarter longer and the borrowed plans a third longer, since models write about two thirds of a
  borrowed plan's words; without that, samples fell under the detector's 150 words. It costs one more
  request, and about twice the tokens, since the profile goes with both.
- The voice rule says nothing when there is no source text.
- Document rewrites borrow the writer's sentences by default; with no shapes to borrow they keep the
  original's.

The built-in profile is still the weaker one, as a profile of no one should be.

### One measured writer instead of the average: worse

Tried the same day: each sample, or each document, written as one of the 1,750 writers the built-in
profile was measured from, the model told how that writer differs from most (their sentence length,
commas, contractions, "I", brackets, slips) and given their own sentence shapes. On
gemini-3.1-flash-lite, the model with quota left that day:

| Built-in profile, finished work | One writer | Their average |
|---|---:|---:|
| Samples from instructions, restyled (9 each) | 93 ± 8, 0 of 9 human | 52 ± 29, 3 of 9 human |
| Claude's report, Writer's sentences (3 trials) | 94 ± 5 | 85 ± 14 |

A finished-work writer drawn at random is as often a science-abstract or news writer as anyone (34
words a sentence, 1.6 commas, long words), and told those numbers the model wrote stiff, formal prose,
the kind the detector is surest about. The average's plain, uneven, concrete prose was the better
instruction, so the built-in profile stays the average.

## Revising what the detector flags, 2026-10-30

Until now the generator was open loop: write, restyle, then score. The page's adaptive mode did close
the loop, but by writing the whole draft again, which brings back the model's own plan each time.
`redteam_batch.revise_flagged` closes it where the detector reads: it scores the sample sentence by
sentence, rewrites only its most AI-like passages (at most half of them) in new sentence plans borrowed
from the writer, told what was flagged (`REVISE`), and keeps a round only if the whole sample then
scores lower, for up to 3 rounds or until it scores under 20%.

Measured with `python -m eval.tune_prompts --style data/styles/alexandria_profile.json --restyle
--revise 3` on gemini-3.1-flash-lite (3.5's daily quota was spent), the same prompts as above, 3 repeats,
scored by the slim detector. *Before* is the same sample after restyle, before any revision.

| Register, prompts | Before: AI likelihood % | Likely human | After: AI likelihood % | Likely human | Words | Tokens per sample |
|---|---:|---:|---:|---:|---:|---:|
| finished work: essay, report, story | 56 ± 32 | 2 of 9 | 21 ± 16 | 7 of 9 | 591 | 63,743 |
| quick notes: review, reply, explanation | 49 ± 30 | 4 of 9 | 18 ± 17 | 8 of 9 | 506 | 56,560 |

The loop and these scores use the slim build (Desklib alone). Scoring the same samples with the full
build stalled on the 8 GB Mac and is still to be done; the full build adds GLYPH and Binoculars, which
the loop never saw.

- **Revising where the detector reads works where restyling the whole draft didn't.** Samples starting
  at 82-95% came down to 9-40% in two or three rounds; a round that made a sample worse was dropped.
- **Most pass in a round or two, a few never do**: 2 of 18 stayed at *Needs review* (63-65%) after 3
  rounds.
- **It does nothing for the style match**, which stays near 0 against the writer's own passages, with or
  without revising: these get past the detector without measurably reading like the writer.
- **It costs** up to one request per round (two with a copy retry) and roughly 2.5 times the tokens.
  The first finished-work run let each revision stretch its passage a third longer (591 words against
  350 asked); revisions now keep their length.

On the Red team page this is **Revise what the detector flags** (2 rounds), for samples from a prompt,
never in public mode. Every revised sample is saved as AI-labelled data with the score after each round
(`revise_rounds`), attack `…+revise4` (2 rounds until 2026-10-31); the 18 above are in `data/essays/tuning_redteam.jsonl`
(`…+revise3`). They are the hard negatives the detector should be refitted on next.

### The full detector, and the stuck samples, 2026-10-31

The same 18 samples were scored on a 16 GB Windows laptop (`deploy/remote_test/`). Slim matched the Mac
sample by sample. The **full build**, whose GLYPH and Binoculars the loop never saw, read them slightly
lower: finished work 19 ± 14%, quick notes 15 ± 13%, 15 likely human, 3 needs review, 0 likely AI. GLYPH
read most at 1-5% and Binoculars none as AI-like; only Desklib flagged the three that stayed borderline.

Those three (and one more at 28%) had lost two or three rounds each: a round was kept or dropped whole,
so one passage that scored worse threw away the ones that helped. `revise_flagged` now keeps each
rewritten passage on its own when the whole text then scores lower. Revised 3 more rounds from where they
stopped (slim, gemini-3.1-flash-lite), with the old loop and the new:

| Sample | Start % | Old loop % | Rounds | New loop % | Rounds |
|---|---:|---:|---:|---:|---:|
| finished 2#1 | 40 | 10 | 1 | 11 | 1 |
| finished 2#2 | 57 | 27 | 3 | 30 | 3 |
| informal 3#1 | 63 | 9 | 3 | 9 | 1 |
| informal 3#2 | 28 | 15 | 1 | 13 | 1 |

- **More rounds is what moved them**: every one ended likely human with either loop. The page now allows
  4 rounds (`REVISE_ROUNDS`, was 2); with 2, 4 of the 18 would have stayed flagged, 1 as *Likely AI*.
  A round only runs while the sample is still over 20%.
- **Keeping passages one by one** got there in fewer rounds once (6 Gemini requests against 8) and
  costs one detector pass per rewritten passage instead of one per round. Four samples can't tell the
  two loops' final scores apart.

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
