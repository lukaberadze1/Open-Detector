# Reference benchmark

966 AI and 278 human texts of known origin, 150 words or more. Written by `python -m eval.reference_benchmark`. Brackets are Wilson 95% intervals, which treat texts as independent, so versions of one source make them too narrow. * marks under 30 texts. Saved samples include text tuned against this detector, so they aren't a held-back test.

## Every detector

| Detector | AI texts | Missed | Human texts | Wrongly called AI | Needs review (human) |
|---|---:|---:|---:|---:|---:|
| slim | 966 | 15% (13% to 17%) | 278 | 2% (1% to 5%) | 22% (17% to 27%) |
| full | 966 | 18% (16% to 21%) | 278 | 1% (1% to 4%) | 22% (17% to 27%) |
| tabularis | 966 | 46% (43% to 49%) | 278 | 4% (2% to 7%) | 0% (0% to 1%) |
| roberta | 966 | 64% (61% to 67%) | 278 | 17% (13% to 22%) | 0% (0% to 1%) |

## Transfer gap on guided samples

How much higher the second detector scores text tuned against our own, in points of AI likelihood. A big gap means the tuning fools the first detector, not AI detection in general.

| Lower detector | Compared with | Texts | Mean gap |
|---|---|---:|---:|
| slim | full | 154 | -2% |
| slim | tabularis | 154 | -8% |
| slim | roberta | 154 | -8% |
| full | slim | 154 | +2% |
| full | tabularis | 154 | -7% |
| full | roberta | 154 | -6% |
| tabularis | slim | 154 | +8% |
| tabularis | full | 154 | +7% |
| tabularis | roberta | 154 | +1% |
| roberta | slim | 154 | +8% |
| roberta | full | 154 | +6% |
| roberta | tabularis | 154 | -1% |

## slim

#### False accusations by genre

| Genre | Texts | Wrongly called AI | Mean AI likelihood |
|---|---:|---:|---:|
| student essay | 100 | 0% (0% to 4%) | 11% |
| hc3 reddit_eli5 | 50 | 0% (0% to 7%) | 3% |
| wp | 26 | 0% (0% to 13%) * | 10% |
| xsum | 25 | 20% (9% to 39%) * | 34% |
| cmv | 20 | 0% (0% to 16%) * | 13% |
| sci | 19 | 5% (1% to 25%) * | 15% |
| eli5 | 13 | 0% (0% to 23%) * | 16% |
| yelp | 5 | 0% (0% to 43%) * | 14% |
| squad | 5 | 0% (0% to 43%) * | 16% |
| cnn | 5 | 0% (0% to 43%) * | 29% |
| dialogsum | 5 | 0% (0% to 43%) * | 20% |
| red-team | 3 | 0% (0% to 56%) * | 3% |
| imdb | 2 | 0% (0% to 66%) * | 1% |

#### Misses by genre

| Genre | Texts | Missed | Mean AI likelihood |
|---|---:|---:|---:|
| red-team | 685 | 21% (18% to 24%) | 64% |
| student essay | 95 | 0% (0% to 4%) | 99% |
| hc3 reddit_eli5 | 50 | 0% (0% to 7%) | 100% |
| wp | 18 | 0% (0% to 18%) * | 100% |
| roct | 16 | 0% (0% to 19%) * | 99% |
| xsum | 13 | 0% (0% to 23%) * | 98% |
| sci | 13 | 0% (0% to 23%) * | 96% |
| squad | 13 | 0% (0% to 23%) * | 97% |
| cmv | 11 | 0% (0% to 26%) * | 93% |
| eli5 | 11 | 0% (0% to 26%) * | 94% |
| tldr | 11 | 0% (0% to 26%) * | 98% |
| imdb | 7 | 0% (0% to 35%) * | 85% |
| yelp | 7 | 0% (0% to 35%) * | 97% |
| hswag | 6 | 0% (0% to 39%) * | 97% |
| cnn | 6 | 0% (0% to 39%) * | 82% |
| pubmed | 3 | 0% (0% to 56%) * | 65% |
| dialogsum | 1 | 0% (0% to 79%) * | 100% |

#### Misses by attack

| Attack | Texts | Missed | Mean AI likelihood |
|---|---:|---:|---:|
| none | 267 | 0% (0% to 1%) | 98% |
| humanizer+academic-control | 40 | 0% (0% to 9%) | 89% |
| humanizer+stealth-rules | 40 | 5% (1% to 17%) | 90% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+best3+drafts3 | 39 | 13% (6% to 27%) | 80% |
| prompt+human_default_profile:finalized+task+restyle | 26 | 15% (6% to 34%) * | 45% |
| prompt+alexandria_profile:notes+task+restyle | 26 | 35% (19% to 54%) * | 31% |
| prompt+alexandria_profile:finalized+task+restyle | 25 | 12% (4% to 30%) * | 57% |
| full_document+alexandria_profile:finalized+writer+t0.8+as-page | 20 | 15% (5% to 36%) * | 72% |
| full_document+plain | 18 | 0% (0% to 18%) * | 98% |
| full_document+alexandria_profile:notes | 18 | 33% (16% to 56%) * | 45% |
| full_document+alexandria_profile:finalized+writer | 18 | 44% (25% to 66%) * | 35% |
| full_document+alexandria_profile:finalized+writer+t0.8+as-page+revise4-sentences | 18 | 44% (25% to 66%) * | 36% |
| math+contextual | 18 | 0% (0% to 18%) * | 76% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+best3 | 15 | 0% (0% to 20%) * | 87% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+best3+revise10-sentences | 15 | 80% (55% to 93%) * | 14% |
| humanizer+cive-bart+sampled | 15 | 0% (0% to 20%) * | 95% |
| humanizer+cive-bart+best8 | 15 | 0% (0% to 20%) * | 97% |
| humanizer+stealth-bart+best8 | 14 | 0% (0% to 22%) * | 99% |
| paraphrase | 14 | 0% (0% to 22%) * | 79% |
| humanizer+stealth-bart+sampled | 13 | 0% (0% to 23%) * | 97% |
| full_document+plain+same | 12 | 42% (19% to 68%) * | 49% |
| full_document+alexandria_profile:finalized+same | 12 | 17% (5% to 45%) * | 50% |
| prompt+human_default_profile:notes+task+restyle | 12 | 17% (5% to 45%) * | 65% |
| math+contextual+t5 | 12 | 0% (0% to 24%) * | 97% |
| full_document+alexandria_profile:notes+same | 11 | 45% (21% to 72%) * | 34% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+best3+drafts3+revise10-sentences | 11 | 82% (52% to 95%) * | 14% |
| prompt+human_default_profile:finalized+task+restyle+persona | 10 | 10% (2% to 40%) * | 83% |
| prompt+alexandria_profile:finalized+challenge+old-voice | 9 | 0% (0% to 30%) * | 86% |
| prompt+alexandria_profile:finalized+task | 9 | 0% (0% to 30%) * | 91% |
| prompt+alexandria_profile:finalized+task+plan | 9 | 0% (0% to 30%) * | 92% |
| prompt+human_default_profile:finalized+challenge+old-voice | 9 | 0% (0% to 30%) * | 81% |
| prompt+alexandria_profile:notes+task | 9 | 11% (2% to 44%) * | 61% |
| prompt+alexandria_profile:finalized+task+restyle+revise3 | 9 | 67% (35% to 88%) * | 21% |
| prompt+alexandria_profile:notes+task+restyle+revise3 | 9 | 67% (35% to 88%) * | 18% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+best3+roundtrip-german | 9 | 11% (2% to 44%) * | 78% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+best3+roundtrip-german+revise10-sentences | 9 | 89% (56% to 98%) * | 17% |
| prompt+human_default_profile:notes+task | 8 | 0% (0% to 32%) * | 92% |
| full_document+alexandria_profile:finalized | 6 | 0% (0% to 39%) * | 65% |
| full_document+plain+writer | 6 | 0% (0% to 39%) * | 94% |
| full_document+alexandria_profile:notes+writer | 6 | 0% (0% to 39%) * | 56% |
| full_document+human_default_profile:finalized+writer | 6 | 0% (0% to 39%) * | 76% |
| full_document+human_default_profile:finalized+same | 6 | 0% (0% to 39%) * | 92% |
| full_document+human_default_profile:notes+writer | 6 | 0% (0% to 39%) * | 72% |
| prompt+alexandria_profile:finalized+task+restyle+plain-words | 6 | 33% (10% to 70%) * | 45% |
| prompt+alexandria_profile:notes+task+restyle+plain-words | 6 | 67% (30% to 90%) * | 19% |
| full_document+alexandria_profile:finalized+writer+revise3-sentences | 6 | 83% (44% to 97%) * | 7% |
| full_document+alexandria_profile:finalized+writer+revise3-windows | 6 | 83% (44% to 97%) * | 7% |
| prompt+human_default_profile:finalized+task | 6 | 0% (0% to 39%) * | 98% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+masked+best3 | 5 | 20% (4% to 62%) * | 59% |
| full_document+alexandria_profile:finalized+writer+t1 | 4 | 0% (0% to 49%) * | 70% |
| full_document+alexandria_profile:finalized+writer+t1+revise3-sentences | 4 | 100% (51% to 100%) * | 7% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+masked+best3+revise10-sentences | 4 | 100% (51% to 100%) * | 7% |
| prompt+alexandria_profile | 3 | 0% (0% to 56%) * | 95% |
| full_document+plain+light | 3 | 100% (44% to 100%) * | 6% |
| full_document+human_default_profile:finalized+writer+persona | 3 | 0% (0% to 56%) * | 94% |
| prompt+alexandria_profile:finalized+task+restyle+examples | 3 | 0% (0% to 56%) * | 93% |
| prompt+alexandria_profile:notes+task+restyle+examples | 3 | 67% (21% to 94%) * | 20% |
| full_document+alexandria_profile:finalized+writer+t1.3 | 3 | 0% (0% to 56%) * | 71% |
| full_document+alexandria_profile:finalized+writer+t1.3+revise3-sentences | 3 | 100% (44% to 100%) * | 10% |
| math+contextual+pegasus | 3 | 0% (0% to 56%) * | 90% |
| filled_answers+alexandria_profile | 2 | 50% (9% to 91%) * | 11% |
| full_document+alexandria_profile+roundtrip-german+clauses+rules+math+revise10 | 2 | 100% (34% to 100%) * | 9% |
| full_document+alexandria_profile+clauses+rules+math+revise10 | 2 | 50% (9% to 91%) * | 27% |
| full_document+alexandria_profile+clauses+rules+math+revise10+masked | 2 | 0% (0% to 66%) * | 68% |
| prompt+pasted_style_profile | 1 | 0% (0% to 79%) * | 98% |
| prompt+alexandria_profile+sub_5_percent_stealth_v3 | 1 | 100% (21% to 100%) * | 2% |
| prompt+alexandria_profile+math+revise10 | 1 | 0% (0% to 79%) * | 96% |
| full_document+human_default_profile:notes+writer+persona | 1 | 0% (0% to 79%) * | 76% |
| full_document+alexandria_profile:finalized+writer+t0.8 | 1 | 0% (0% to 79%) * | 98% |
| full_document+alexandria_profile:finalized+writer+t0.8+revise3-sentences | 1 | 0% (0% to 79%) * | 16% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page | 1 | 0% (0% to 79%) * | 100% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+revise10-sentences | 1 | 100% (21% to 100%) * | 8% |

## full

#### False accusations by genre

| Genre | Texts | Wrongly called AI | Mean AI likelihood |
|---|---:|---:|---:|
| student essay | 100 | 0% (0% to 4%) | 11% |
| hc3 reddit_eli5 | 50 | 0% (0% to 7%) | 3% |
| wp | 26 | 0% (0% to 13%) * | 10% |
| xsum | 25 | 12% (4% to 30%) * | 30% |
| cmv | 20 | 0% (0% to 16%) * | 15% |
| sci | 19 | 5% (1% to 25%) * | 19% |
| eli5 | 13 | 0% (0% to 23%) * | 13% |
| yelp | 5 | 0% (0% to 43%) * | 14% |
| squad | 5 | 0% (0% to 43%) * | 18% |
| cnn | 5 | 0% (0% to 43%) * | 21% |
| dialogsum | 5 | 0% (0% to 43%) * | 18% |
| red-team | 3 | 0% (0% to 56%) * | 2% |
| imdb | 2 | 0% (0% to 66%) * | 1% |

#### Misses by genre

| Genre | Texts | Missed | Mean AI likelihood |
|---|---:|---:|---:|
| red-team | 685 | 26% (22% to 29%) | 61% |
| student essay | 95 | 0% (0% to 4%) | 100% |
| hc3 reddit_eli5 | 50 | 0% (0% to 7%) | 100% |
| wp | 18 | 0% (0% to 18%) * | 100% |
| roct | 16 | 0% (0% to 19%) * | 98% |
| xsum | 13 | 0% (0% to 23%) * | 98% |
| sci | 13 | 0% (0% to 23%) * | 98% |
| squad | 13 | 0% (0% to 23%) * | 97% |
| cmv | 11 | 0% (0% to 26%) * | 95% |
| eli5 | 11 | 0% (0% to 26%) * | 97% |
| tldr | 11 | 0% (0% to 26%) * | 99% |
| imdb | 7 | 0% (0% to 35%) * | 89% |
| yelp | 7 | 0% (0% to 35%) * | 97% |
| hswag | 6 | 0% (0% to 39%) * | 97% |
| cnn | 6 | 0% (0% to 39%) * | 83% |
| pubmed | 3 | 0% (0% to 56%) * | 69% |
| dialogsum | 1 | 0% (0% to 79%) * | 100% |

#### Misses by attack

| Attack | Texts | Missed | Mean AI likelihood |
|---|---:|---:|---:|
| none | 267 | 0% (0% to 1%) | 99% |
| humanizer+academic-control | 40 | 0% (0% to 9%) | 90% |
| humanizer+stealth-rules | 40 | 2% (0% to 13%) | 91% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+best3+drafts3 | 39 | 13% (6% to 27%) | 78% |
| prompt+human_default_profile:finalized+task+restyle | 26 | 38% (22% to 57%) * | 35% |
| prompt+alexandria_profile:notes+task+restyle | 26 | 58% (39% to 74%) * | 21% |
| prompt+alexandria_profile:finalized+task+restyle | 25 | 20% (9% to 39%) * | 46% |
| full_document+alexandria_profile:finalized+writer+t0.8+as-page | 20 | 20% (8% to 42%) * | 70% |
| full_document+plain | 18 | 0% (0% to 18%) * | 98% |
| full_document+alexandria_profile:notes | 18 | 33% (16% to 56%) * | 37% |
| full_document+alexandria_profile:finalized+writer | 18 | 50% (29% to 71%) * | 29% |
| full_document+alexandria_profile:finalized+writer+t0.8+as-page+revise4-sentences | 18 | 44% (25% to 66%) * | 33% |
| math+contextual | 18 | 0% (0% to 18%) * | 79% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+best3 | 15 | 13% (4% to 38%) * | 86% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+best3+revise10-sentences | 15 | 80% (55% to 93%) * | 12% |
| humanizer+cive-bart+sampled | 15 | 7% (1% to 30%) * | 94% |
| humanizer+cive-bart+best8 | 15 | 0% (0% to 20%) * | 96% |
| humanizer+stealth-bart+best8 | 14 | 0% (0% to 22%) * | 99% |
| paraphrase | 14 | 0% (0% to 22%) * | 82% |
| humanizer+stealth-bart+sampled | 13 | 0% (0% to 23%) * | 98% |
| full_document+plain+same | 12 | 50% (25% to 75%) * | 45% |
| full_document+alexandria_profile:finalized+same | 12 | 25% (9% to 53%) * | 39% |
| prompt+human_default_profile:notes+task+restyle | 12 | 33% (14% to 61%) * | 52% |
| math+contextual+t5 | 12 | 0% (0% to 24%) * | 99% |
| full_document+alexandria_profile:notes+same | 11 | 45% (21% to 72%) * | 28% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+best3+drafts3+revise10-sentences | 11 | 91% (62% to 98%) * | 11% |
| prompt+human_default_profile:finalized+task+restyle+persona | 10 | 10% (2% to 40%) * | 77% |
| prompt+alexandria_profile:finalized+challenge+old-voice | 9 | 0% (0% to 30%) * | 83% |
| prompt+alexandria_profile:finalized+task | 9 | 0% (0% to 30%) * | 87% |
| prompt+alexandria_profile:finalized+task+plan | 9 | 0% (0% to 30%) * | 91% |
| prompt+human_default_profile:finalized+challenge+old-voice | 9 | 0% (0% to 30%) * | 73% |
| prompt+alexandria_profile:notes+task | 9 | 11% (2% to 44%) * | 53% |
| prompt+alexandria_profile:finalized+task+restyle+revise3 | 9 | 56% (27% to 81%) * | 19% |
| prompt+alexandria_profile:notes+task+restyle+revise3 | 9 | 78% (45% to 94%) * | 15% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+best3+roundtrip-german | 9 | 22% (6% to 55%) * | 76% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+best3+roundtrip-german+revise10-sentences | 9 | 89% (56% to 98%) * | 16% |
| prompt+human_default_profile:notes+task | 8 | 0% (0% to 32%) * | 87% |
| full_document+alexandria_profile:finalized | 6 | 0% (0% to 39%) * | 57% |
| full_document+plain+writer | 6 | 0% (0% to 39%) * | 89% |
| full_document+alexandria_profile:notes+writer | 6 | 0% (0% to 39%) * | 44% |
| full_document+human_default_profile:finalized+writer | 6 | 0% (0% to 39%) * | 66% |
| full_document+human_default_profile:finalized+same | 6 | 0% (0% to 39%) * | 88% |
| full_document+human_default_profile:notes+writer | 6 | 0% (0% to 39%) * | 59% |
| prompt+alexandria_profile:finalized+task+restyle+plain-words | 6 | 50% (19% to 81%) * | 35% |
| prompt+alexandria_profile:notes+task+restyle+plain-words | 6 | 83% (44% to 97%) * | 14% |
| full_document+alexandria_profile:finalized+writer+revise3-sentences | 6 | 100% (61% to 100%) * | 4% |
| full_document+alexandria_profile:finalized+writer+revise3-windows | 6 | 100% (61% to 100%) * | 4% |
| prompt+human_default_profile:finalized+task | 6 | 0% (0% to 39%) * | 97% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+masked+best3 | 5 | 20% (4% to 62%) * | 59% |
| full_document+alexandria_profile:finalized+writer+t1 | 4 | 0% (0% to 49%) * | 60% |
| full_document+alexandria_profile:finalized+writer+t1+revise3-sentences | 4 | 100% (51% to 100%) * | 4% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+masked+best3+revise10-sentences | 4 | 100% (51% to 100%) * | 7% |
| prompt+alexandria_profile | 3 | 0% (0% to 56%) * | 95% |
| full_document+plain+light | 3 | 100% (44% to 100%) * | 4% |
| full_document+human_default_profile:finalized+writer+persona | 3 | 0% (0% to 56%) * | 93% |
| prompt+alexandria_profile:finalized+task+restyle+examples | 3 | 0% (0% to 56%) * | 93% |
| prompt+alexandria_profile:notes+task+restyle+examples | 3 | 67% (21% to 94%) * | 13% |
| full_document+alexandria_profile:finalized+writer+t1.3 | 3 | 0% (0% to 56%) * | 68% |
| full_document+alexandria_profile:finalized+writer+t1.3+revise3-sentences | 3 | 100% (44% to 100%) * | 8% |
| math+contextual+pegasus | 3 | 0% (0% to 56%) * | 94% |
| filled_answers+alexandria_profile | 2 | 100% (34% to 100%) * | 7% |
| full_document+alexandria_profile+roundtrip-german+clauses+rules+math+revise10 | 2 | 100% (34% to 100%) * | 9% |
| full_document+alexandria_profile+clauses+rules+math+revise10 | 2 | 50% (9% to 91%) * | 24% |
| full_document+alexandria_profile+clauses+rules+math+revise10+masked | 2 | 0% (0% to 66%) * | 73% |
| prompt+pasted_style_profile | 1 | 0% (0% to 79%) * | 96% |
| prompt+alexandria_profile+sub_5_percent_stealth_v3 | 1 | 100% (21% to 100%) * | 2% |
| prompt+alexandria_profile+math+revise10 | 1 | 0% (0% to 79%) * | 96% |
| full_document+human_default_profile:notes+writer+persona | 1 | 0% (0% to 79%) * | 65% |
| full_document+alexandria_profile:finalized+writer+t0.8 | 1 | 0% (0% to 79%) * | 97% |
| full_document+alexandria_profile:finalized+writer+t0.8+revise3-sentences | 1 | 100% (21% to 100%) * | 12% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page | 1 | 0% (0% to 79%) * | 100% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+revise10-sentences | 1 | 100% (21% to 100%) * | 6% |

## tabularis

#### False accusations by genre

| Genre | Texts | Wrongly called AI | Mean AI likelihood |
|---|---:|---:|---:|
| student essay | 100 | 1% (0% to 5%) | 8% |
| hc3 reddit_eli5 | 50 | 0% (0% to 7%) | 0% |
| wp | 26 | 0% (0% to 13%) * | 4% |
| xsum | 25 | 16% (6% to 35%) * | 23% |
| cmv | 20 | 10% (3% to 30%) * | 13% |
| sci | 19 | 5% (1% to 25%) * | 6% |
| eli5 | 13 | 0% (0% to 23%) * | 6% |
| yelp | 5 | 0% (0% to 43%) * | 6% |
| squad | 5 | 20% (4% to 62%) * | 16% |
| cnn | 5 | 20% (4% to 62%) * | 39% |
| dialogsum | 5 | 20% (4% to 62%) * | 21% |
| red-team | 3 | 0% (0% to 56%) * | 0% |
| imdb | 2 | 0% (0% to 66%) * | 1% |

#### Misses by genre

| Genre | Texts | Missed | Mean AI likelihood |
|---|---:|---:|---:|
| red-team | 685 | 62% (58% to 66%) | 40% |
| student essay | 95 | 1% (0% to 6%) | 97% |
| hc3 reddit_eli5 | 50 | 0% (0% to 7%) | 99% |
| wp | 18 | 0% (0% to 18%) * | 90% |
| roct | 16 | 19% (7% to 43%) * | 77% |
| xsum | 13 | 0% (0% to 23%) * | 96% |
| sci | 13 | 8% (1% to 33%) * | 90% |
| squad | 13 | 15% (4% to 42%) * | 80% |
| cmv | 11 | 0% (0% to 26%) * | 95% |
| eli5 | 11 | 9% (2% to 38%) * | 86% |
| tldr | 11 | 18% (5% to 48%) * | 79% |
| imdb | 7 | 14% (3% to 51%) * | 74% |
| yelp | 7 | 14% (3% to 51%) * | 84% |
| hswag | 6 | 0% (0% to 39%) * | 95% |
| cnn | 6 | 33% (10% to 70%) * | 58% |
| pubmed | 3 | 67% (21% to 94%) * | 31% |
| dialogsum | 1 | 0% (0% to 79%) * | 99% |

#### Misses by attack

| Attack | Texts | Missed | Mean AI likelihood |
|---|---:|---:|---:|
| none | 267 | 4% (3% to 8%) | 92% |
| humanizer+academic-control | 40 | 10% (4% to 23%) | 87% |
| humanizer+stealth-rules | 40 | 8% (3% to 20%) | 85% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+best3+drafts3 | 39 | 41% (27% to 57%) | 61% |
| prompt+human_default_profile:finalized+task+restyle | 26 | 100% (87% to 100%) * | 4% |
| prompt+alexandria_profile:notes+task+restyle | 26 | 92% (76% to 98%) * | 12% |
| prompt+alexandria_profile:finalized+task+restyle | 25 | 92% (75% to 98%) * | 11% |
| full_document+alexandria_profile:finalized+writer+t0.8+as-page | 20 | 40% (22% to 61%) * | 55% |
| full_document+plain | 18 | 67% (44% to 84%) * | 37% |
| full_document+alexandria_profile:notes | 18 | 100% (82% to 100%) * | 8% |
| full_document+alexandria_profile:finalized+writer | 18 | 100% (82% to 100%) * | 10% |
| full_document+alexandria_profile:finalized+writer+t0.8+as-page+revise4-sentences | 18 | 72% (49% to 88%) * | 35% |
| math+contextual | 18 | 28% (12% to 51%) * | 71% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+best3 | 15 | 20% (7% to 45%) * | 71% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+best3+revise10-sentences | 15 | 80% (55% to 93%) * | 25% |
| humanizer+cive-bart+sampled | 15 | 20% (7% to 45%) * | 79% |
| humanizer+cive-bart+best8 | 15 | 20% (7% to 45%) * | 76% |
| humanizer+stealth-bart+best8 | 14 | 7% (1% to 31%) * | 80% |
| paraphrase | 14 | 29% (12% to 55%) * | 61% |
| humanizer+stealth-bart+sampled | 13 | 8% (1% to 33%) * | 87% |
| full_document+plain+same | 12 | 75% (47% to 91%) * | 18% |
| full_document+alexandria_profile:finalized+same | 12 | 100% (76% to 100%) * | 6% |
| prompt+human_default_profile:notes+task+restyle | 12 | 83% (55% to 95%) * | 33% |
| math+contextual+t5 | 12 | 0% (0% to 24%) * | 92% |
| full_document+alexandria_profile:notes+same | 11 | 100% (74% to 100%) * | 2% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+best3+drafts3+revise10-sentences | 11 | 82% (52% to 95%) * | 22% |
| prompt+human_default_profile:finalized+task+restyle+persona | 10 | 60% (31% to 83%) * | 49% |
| prompt+alexandria_profile:finalized+challenge+old-voice | 9 | 100% (70% to 100%) * | 10% |
| prompt+alexandria_profile:finalized+task | 9 | 89% (56% to 98%) * | 16% |
| prompt+alexandria_profile:finalized+task+plan | 9 | 89% (56% to 98%) * | 18% |
| prompt+human_default_profile:finalized+challenge+old-voice | 9 | 100% (70% to 100%) * | 13% |
| prompt+alexandria_profile:notes+task | 9 | 78% (45% to 94%) * | 32% |
| prompt+alexandria_profile:finalized+task+restyle+revise3 | 9 | 100% (70% to 100%) * | 9% |
| prompt+alexandria_profile:notes+task+restyle+revise3 | 9 | 89% (56% to 98%) * | 16% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+best3+roundtrip-german | 9 | 44% (19% to 73%) * | 52% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+best3+roundtrip-german+revise10-sentences | 9 | 89% (56% to 98%) * | 25% |
| prompt+human_default_profile:notes+task | 8 | 38% (14% to 69%) * | 62% |
| full_document+alexandria_profile:finalized | 6 | 100% (61% to 100%) * | 8% |
| full_document+plain+writer | 6 | 83% (44% to 97%) * | 36% |
| full_document+alexandria_profile:notes+writer | 6 | 100% (61% to 100%) * | 13% |
| full_document+human_default_profile:finalized+writer | 6 | 100% (61% to 100%) * | 8% |
| full_document+human_default_profile:finalized+same | 6 | 100% (61% to 100%) * | 9% |
| full_document+human_default_profile:notes+writer | 6 | 100% (61% to 100%) * | 16% |
| prompt+alexandria_profile:finalized+task+restyle+plain-words | 6 | 100% (61% to 100%) * | 4% |
| prompt+alexandria_profile:notes+task+restyle+plain-words | 6 | 100% (61% to 100%) * | 7% |
| full_document+alexandria_profile:finalized+writer+revise3-sentences | 6 | 100% (61% to 100%) * | 1% |
| full_document+alexandria_profile:finalized+writer+revise3-windows | 6 | 100% (61% to 100%) * | 3% |
| prompt+human_default_profile:finalized+task | 6 | 0% (0% to 39%) * | 96% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+masked+best3 | 5 | 20% (4% to 62%) * | 59% |
| full_document+alexandria_profile:finalized+writer+t1 | 4 | 100% (51% to 100%) * | 2% |
| full_document+alexandria_profile:finalized+writer+t1+revise3-sentences | 4 | 100% (51% to 100%) * | 1% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+masked+best3+revise10-sentences | 4 | 50% (15% to 85%) * | 49% |
| prompt+alexandria_profile | 3 | 33% (6% to 79%) * | 54% |
| full_document+plain+light | 3 | 100% (44% to 100%) * | 0% |
| full_document+human_default_profile:finalized+writer+persona | 3 | 100% (44% to 100%) * | 46% |
| prompt+alexandria_profile:finalized+task+restyle+examples | 3 | 100% (44% to 100%) * | 28% |
| prompt+alexandria_profile:notes+task+restyle+examples | 3 | 100% (44% to 100%) * | 14% |
| full_document+alexandria_profile:finalized+writer+t1.3 | 3 | 100% (44% to 100%) * | 23% |
| full_document+alexandria_profile:finalized+writer+t1.3+revise3-sentences | 3 | 100% (44% to 100%) * | 21% |
| math+contextual+pegasus | 3 | 0% (0% to 56%) * | 98% |
| filled_answers+alexandria_profile | 2 | 100% (34% to 100%) * | 6% |
| full_document+alexandria_profile+roundtrip-german+clauses+rules+math+revise10 | 2 | 100% (34% to 100%) * | 2% |
| full_document+alexandria_profile+clauses+rules+math+revise10 | 2 | 100% (34% to 100%) * | 15% |
| full_document+alexandria_profile+clauses+rules+math+revise10+masked | 2 | 50% (9% to 91%) * | 51% |
| prompt+pasted_style_profile | 1 | 100% (21% to 100%) * | 40% |
| prompt+alexandria_profile+sub_5_percent_stealth_v3 | 1 | 100% (21% to 100%) * | 28% |
| prompt+alexandria_profile+math+revise10 | 1 | 100% (21% to 100%) * | 0% |
| full_document+human_default_profile:notes+writer+persona | 1 | 100% (21% to 100%) * | 26% |
| full_document+alexandria_profile:finalized+writer+t0.8 | 1 | 0% (0% to 79%) * | 68% |
| full_document+alexandria_profile:finalized+writer+t0.8+revise3-sentences | 1 | 100% (21% to 100%) * | 30% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page | 1 | 0% (0% to 79%) * | 53% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+revise10-sentences | 1 | 100% (21% to 100%) * | 4% |

## roberta

#### False accusations by genre

| Genre | Texts | Wrongly called AI | Mean AI likelihood |
|---|---:|---:|---:|
| student essay | 100 | 26% (18% to 35%) | 28% |
| hc3 reddit_eli5 | 50 | 0% (0% to 7%) | 0% |
| wp | 26 | 15% (6% to 34%) * | 19% |
| xsum | 25 | 36% (20% to 55%) * | 47% |
| cmv | 20 | 0% (0% to 16%) * | 1% |
| sci | 19 | 5% (1% to 25%) * | 5% |
| eli5 | 13 | 23% (8% to 50%) * | 27% |
| yelp | 5 | 0% (0% to 43%) * | 1% |
| squad | 5 | 0% (0% to 43%) * | 8% |
| cnn | 5 | 20% (4% to 62%) * | 13% |
| dialogsum | 5 | 60% (23% to 88%) * | 59% |
| red-team | 3 | 33% (6% to 79%) * | 31% |
| imdb | 2 | 0% (0% to 66%) * | 10% |

#### Misses by genre

| Genre | Texts | Missed | Mean AI likelihood |
|---|---:|---:|---:|
| red-team | 685 | 77% (74% to 80%) | 25% |
| student essay | 95 | 11% (6% to 18%) | 86% |
| hc3 reddit_eli5 | 50 | 0% (0% to 7%) | 100% |
| wp | 18 | 50% (29% to 71%) * | 57% |
| roct | 16 | 56% (33% to 77%) * | 51% |
| xsum | 13 | 15% (4% to 42%) * | 81% |
| sci | 13 | 92% (67% to 99%) * | 12% |
| squad | 13 | 85% (58% to 96%) * | 28% |
| cmv | 11 | 82% (52% to 95%) * | 20% |
| eli5 | 11 | 64% (35% to 85%) * | 39% |
| tldr | 11 | 45% (21% to 72%) * | 59% |
| imdb | 7 | 14% (3% to 51%) * | 70% |
| yelp | 7 | 86% (49% to 97%) * | 20% |
| hswag | 6 | 67% (30% to 90%) * | 39% |
| cnn | 6 | 33% (10% to 70%) * | 59% |
| pubmed | 3 | 67% (21% to 94%) * | 44% |
| dialogsum | 1 | 100% (21% to 100%) * | 14% |

#### Misses by attack

| Attack | Texts | Missed | Mean AI likelihood |
|---|---:|---:|---:|
| none | 267 | 32% (27% to 38%) | 69% |
| humanizer+academic-control | 40 | 55% (40% to 69%) | 48% |
| humanizer+stealth-rules | 40 | 52% (37% to 67%) | 48% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+best3+drafts3 | 39 | 62% (46% to 75%) | 41% |
| prompt+human_default_profile:finalized+task+restyle | 26 | 92% (76% to 98%) * | 12% |
| prompt+alexandria_profile:notes+task+restyle | 26 | 100% (87% to 100%) * | 0% |
| prompt+alexandria_profile:finalized+task+restyle | 25 | 92% (75% to 98%) * | 7% |
| full_document+alexandria_profile:finalized+writer+t0.8+as-page | 20 | 70% (48% to 85%) * | 35% |
| full_document+plain | 18 | 100% (82% to 100%) * | 5% |
| full_document+alexandria_profile:notes | 18 | 94% (74% to 99%) * | 3% |
| full_document+alexandria_profile:finalized+writer | 18 | 94% (74% to 99%) * | 10% |
| full_document+alexandria_profile:finalized+writer+t0.8+as-page+revise4-sentences | 18 | 72% (49% to 88%) * | 35% |
| math+contextual | 18 | 44% (25% to 66%) * | 57% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+best3 | 15 | 73% (48% to 89%) * | 30% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+best3+revise10-sentences | 15 | 80% (55% to 93%) * | 26% |
| humanizer+cive-bart+sampled | 15 | 27% (11% to 52%) * | 71% |
| humanizer+cive-bart+best8 | 15 | 47% (25% to 70%) * | 53% |
| humanizer+stealth-bart+best8 | 14 | 86% (60% to 96%) * | 15% |
| paraphrase | 14 | 29% (12% to 55%) * | 60% |
| humanizer+stealth-bart+sampled | 13 | 62% (36% to 82%) * | 37% |
| full_document+plain+same | 12 | 100% (76% to 100%) * | 0% |
| full_document+alexandria_profile:finalized+same | 12 | 100% (76% to 100%) * | 10% |
| prompt+human_default_profile:notes+task+restyle | 12 | 100% (76% to 100%) * | 4% |
| math+contextual+t5 | 12 | 42% (19% to 68%) * | 58% |
| full_document+alexandria_profile:notes+same | 11 | 100% (74% to 100%) * | 1% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+best3+drafts3+revise10-sentences | 11 | 73% (43% to 90%) * | 25% |
| prompt+human_default_profile:finalized+task+restyle+persona | 10 | 60% (31% to 83%) * | 35% |
| prompt+alexandria_profile:finalized+challenge+old-voice | 9 | 100% (70% to 100%) * | 6% |
| prompt+alexandria_profile:finalized+task | 9 | 100% (70% to 100%) * | 3% |
| prompt+alexandria_profile:finalized+task+plan | 9 | 100% (70% to 100%) * | 1% |
| prompt+human_default_profile:finalized+challenge+old-voice | 9 | 67% (35% to 88%) * | 32% |
| prompt+alexandria_profile:notes+task | 9 | 100% (70% to 100%) * | 0% |
| prompt+alexandria_profile:finalized+task+restyle+revise3 | 9 | 100% (70% to 100%) * | 10% |
| prompt+alexandria_profile:notes+task+restyle+revise3 | 9 | 100% (70% to 100%) * | 0% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+best3+roundtrip-german | 9 | 56% (27% to 81%) * | 52% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+best3+roundtrip-german+revise10-sentences | 9 | 44% (19% to 73%) * | 52% |
| prompt+human_default_profile:notes+task | 8 | 100% (68% to 100%) * | 5% |
| full_document+alexandria_profile:finalized | 6 | 100% (61% to 100%) * | 5% |
| full_document+plain+writer | 6 | 100% (61% to 100%) * | 2% |
| full_document+alexandria_profile:notes+writer | 6 | 100% (61% to 100%) * | 1% |
| full_document+human_default_profile:finalized+writer | 6 | 100% (61% to 100%) * | 5% |
| full_document+human_default_profile:finalized+same | 6 | 100% (61% to 100%) * | 10% |
| full_document+human_default_profile:notes+writer | 6 | 100% (61% to 100%) * | 2% |
| prompt+alexandria_profile:finalized+task+restyle+plain-words | 6 | 100% (61% to 100%) * | 12% |
| prompt+alexandria_profile:notes+task+restyle+plain-words | 6 | 100% (61% to 100%) * | 0% |
| full_document+alexandria_profile:finalized+writer+revise3-sentences | 6 | 67% (30% to 90%) * | 27% |
| full_document+alexandria_profile:finalized+writer+revise3-windows | 6 | 83% (44% to 97%) * | 14% |
| prompt+human_default_profile:finalized+task | 6 | 100% (61% to 100%) * | 33% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+masked+best3 | 5 | 60% (23% to 88%) * | 40% |
| full_document+alexandria_profile:finalized+writer+t1 | 4 | 75% (30% to 95%) * | 25% |
| full_document+alexandria_profile:finalized+writer+t1+revise3-sentences | 4 | 75% (30% to 95%) * | 44% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+masked+best3+revise10-sentences | 4 | 75% (30% to 95%) * | 30% |
| prompt+alexandria_profile | 3 | 67% (21% to 94%) * | 37% |
| full_document+plain+light | 3 | 100% (44% to 100%) * | 0% |
| full_document+human_default_profile:finalized+writer+persona | 3 | 33% (6% to 79%) * | 71% |
| prompt+alexandria_profile:finalized+task+restyle+examples | 3 | 100% (44% to 100%) * | 3% |
| prompt+alexandria_profile:notes+task+restyle+examples | 3 | 100% (44% to 100%) * | 4% |
| full_document+alexandria_profile:finalized+writer+t1.3 | 3 | 100% (44% to 100%) * | 20% |
| full_document+alexandria_profile:finalized+writer+t1.3+revise3-sentences | 3 | 100% (44% to 100%) * | 20% |
| math+contextual+pegasus | 3 | 33% (6% to 79%) * | 66% |
| filled_answers+alexandria_profile | 2 | 100% (34% to 100%) * | 0% |
| full_document+alexandria_profile+roundtrip-german+clauses+rules+math+revise10 | 2 | 50% (9% to 91%) * | 74% |
| full_document+alexandria_profile+clauses+rules+math+revise10 | 2 | 100% (34% to 100%) * | 5% |
| full_document+alexandria_profile+clauses+rules+math+revise10+masked | 2 | 100% (34% to 100%) * | 5% |
| prompt+pasted_style_profile | 1 | 100% (21% to 100%) * | 0% |
| prompt+alexandria_profile+sub_5_percent_stealth_v3 | 1 | 100% (21% to 100%) * | 0% |
| prompt+alexandria_profile+math+revise10 | 1 | 0% (0% to 79%) * | 99% |
| full_document+human_default_profile:notes+writer+persona | 1 | 100% (21% to 100%) * | 2% |
| full_document+alexandria_profile:finalized+writer+t0.8 | 1 | 0% (0% to 79%) * | 66% |
| full_document+alexandria_profile:finalized+writer+t0.8+revise3-sentences | 1 | 100% (21% to 100%) * | 39% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page | 1 | 0% (0% to 79%) * | 100% |
| full_document+alexandria_profile:finalized+writer+passages+patience+t0.8+as-page+revise10-sentences | 1 | 0% (0% to 79%) * | 100% |

