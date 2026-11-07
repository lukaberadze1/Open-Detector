# Public humanizers against the detector

AI texts of MAGE no combiner was fitted on, through each humanizer, scored with the slim detector and its shipped thresholds. Written by `python -m eval.humanizer_attack`. Missed is Likely human; brackets are Wilson 95% intervals. None of these was guided by this detector.

| Humanizer | Texts | Before % | After % | Caught | Missed | Words kept % | Names, numbers kept % | Length |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| academic-control | 40 | 95 | 89 | 82% (68% to 91%) | 0% (0% to 9%) | 98 | 100 | x1.04 |
| stealth-rules | 40 | 95 | 90 | 88% (74% to 95%) | 5% (1% to 17%) | 98 | 96 | x1.01 |
| cive-bart+sampled | 15 | 99 | 95 | 93% (70% to 99%) | 0% (0% to 20%) | 84 | 80 | x1.09 |
| stealth-bart+sampled | 13 | 100 | 97 | 92% (67% to 99%) | 0% (0% to 23%) | 49 | 35 | x1.12 |
| cive-bart+best8 | 15 | 99 | 97 | 93% (70% to 99%) | 0% (0% to 20%) | 85 | 84 | x1.03 |
| stealth-bart+best8 | 14 | 100 | 99 | 100% (78% to 100%) | 0% (0% to 22%) | 60 | 36 | x1.22 |
