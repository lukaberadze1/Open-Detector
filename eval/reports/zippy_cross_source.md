# Cross-source ZipPy evaluation (2026-11-03)

Exploratory document-grouped evaluation. It does not change the production combiner.
MAGE provides no document IDs; identical normalized texts are grouped by hash. RAID uses `group_key` from its loader.
Rows after the 150-word minimum: 1859; bootstrap replicates: 1000; seed: 0.

A logistic model is fitted on training groups, calibrated on separate human groups to an empirical 1% FPR threshold,
then evaluated once on test groups. Confidence intervals use document-group bootstrap resampling.

| Variant | Threshold | Calibration FPR | Test FPR | Test TPR | AUROC (95% CI) | Balanced accuracy (95% CI) |
|---|---:|---:|---:|---:|---:|---:|
| ZipPy | 0.801932 | 0.6% | 1.4% | 4.1% | 0.725 (0.675–0.773) | 0.513 (0.499–0.529) |
| ZipPy + deterministic features | 0.812565 | 0.6% | 1.0% | 3.6% | 0.723 (0.672–0.771) | 0.513 (0.501–0.528) |

## Split sizes

| Split | Documents | Source groups | Human | AI |
|---|---:|---:|---:|---:|
| train | 1090 | 1074 | 506 | 584 |
| calibration | 365 | 357 | 177 | 188 |
| test | 404 | 398 | 210 | 194 |

## Test results by dataset and domain

| Dataset | Domain | Human | AI | FPR | TPR |
|---|---|---:|---:|---:|---:|

### ZipPy

| mage | cmv | 16 | 12 | 0.0% | 0.0% |
| mage | cnn | 8 | 7 | 0.0% | 0.0% |
| mage | dialogsum | 5 | 6 | 0.0% | 0.0% |
| mage | eli5 | 18 | 12 | 0.0% | 8.3% |
| mage | hswag | 0 | 7 | – | 0.0% |
| mage | imdb | 4 | 9 | 0.0% | 11.1% |
| mage | pubmed | 0 | 1 | – | 0.0% |
| mage | roct | 0 | 25 | – | 4.0% |
| mage | sci | 20 | 18 | 15.0% | 16.7% |
| mage | squad | 10 | 18 | 0.0% | 0.0% |
| mage | tldr | 0 | 17 | – | 0.0% |
| mage | wp | 47 | 24 | 0.0% | 0.0% |
| mage | xsum | 42 | 19 | 0.0% | 5.3% |
| mage | yelp | 9 | 8 | 0.0% | 0.0% |
| raid | abstracts | 4 | 0 | 0.0% | – |
| raid | books | 7 | 0 | 0.0% | – |
| raid | news | 11 | 7 | 0.0% | 14.3% |
| raid | recipes | 9 | 2 | 0.0% | 0.0% |
| raid | wiki | 0 | 2 | – | 0.0% |

### ZipPy + deterministic features

| mage | cmv | 16 | 12 | 0.0% | 0.0% |
| mage | cnn | 8 | 7 | 0.0% | 0.0% |
| mage | dialogsum | 5 | 6 | 0.0% | 0.0% |
| mage | eli5 | 18 | 12 | 0.0% | 8.3% |
| mage | hswag | 0 | 7 | – | 0.0% |
| mage | imdb | 4 | 9 | 0.0% | 0.0% |
| mage | pubmed | 0 | 1 | – | 0.0% |
| mage | roct | 0 | 25 | – | 4.0% |
| mage | sci | 20 | 18 | 10.0% | 11.1% |
| mage | squad | 10 | 18 | 0.0% | 0.0% |
| mage | tldr | 0 | 17 | – | 5.9% |
| mage | wp | 47 | 24 | 0.0% | 0.0% |
| mage | xsum | 42 | 19 | 0.0% | 5.3% |
| mage | yelp | 9 | 8 | 0.0% | 0.0% |
| raid | abstracts | 4 | 0 | 0.0% | – |
| raid | books | 7 | 0 | 0.0% | – |
| raid | news | 11 | 7 | 0.0% | 14.3% |
| raid | recipes | 9 | 2 | 0.0% | 0.0% |
| raid | wiki | 0 | 2 | – | 0.0% |

## Features and limits

ZipPy uses `zippy_lzma` and `zippy_zlib`. The combined variant adds sentence-length variation (`burst_cv`, `burst_adj`) and a deterministic repeated word-bigram rate.
The 1% threshold is empirical; with fewer than 100 human calibration groups the script stops. The CI describes test-group resampling, not uncertainty from new domains or data collection.
No result here is an entropy measurement or a production calibration.
