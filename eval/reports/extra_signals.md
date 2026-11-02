# Burstiness, Fast-DetectGPT, raw perplexity and RoBERTa as combiner features (2026-11-02)

1,063 RAID and MAGE texts of 150+ words of prose, averaged over 20 splits by source document, with each combiner's
thresholds set as in `eval/fit_combiner.py`. Red team: 250 saved AI samples, no human ones, scored through the slim
variants only. Fast-DetectGPT and perplexity use Qwen2.5-0.5B. The shipped combiners were not changed.

| Combiner | AUROC | AI caught at 1% FPR | Caught at its threshold | Human flagged | Paraphrase caught | Red team caught |
|---|---:|---:|---:|---:|---:|---:|
| Slim (Desklib) | 0.9822 | 84.7% | 82.7% | 0.92% | 38% | 24.4% |
| + Fast-DetectGPT | 0.9856 | 84.4% | 80.6% | 0.90% | 39% | 17.6% |
| + perplexity | 0.9822 | 84.8% | 82.6% | 1.02% | 38% | 22.5% |
| + RoBERTa (OpenAI GPT-2) | 0.9862 | 82.6% | 81.4% | 0.90% | 40% | 11.2% |
| + RoBERTa (ChatGPT) | 0.9822 | 84.6% | 82.0% | 0.92% | 38% | 21.8% |
| + burstiness | 0.9818 | 85.0% | 83.0% | 0.98% | 38% | 24.9% |
| Full | 0.9856 | 85.1% | 83.5% | 0.75% | 42% | not run |
| + Fast-DetectGPT | 0.9857 | 84.8% | 82.7% | 0.75% | 42% | not run |
| + perplexity | 0.9867 | 84.7% | 84.0% | 0.86% | 42% | not run |
| + RoBERTa (OpenAI GPT-2) | 0.9874 | 82.5% | 82.0% | 0.80% | 43% | not run |
| + RoBERTa (ChatGPT) | 0.9855 | 84.8% | 83.6% | 0.75% | 42% | not run |
| + burstiness | 0.9852 | 84.8% | 83.1% | 0.70% | 41% | not run |
| + all five | 0.9889 | 82.9% | 80.5% | 0.98% | 42% | not run |

Single signals, AUROC on one test split: burstiness 0.55, Fast-DetectGPT 0.69, RoBERTa (ChatGPT) 0.72,
perplexity 0.76, RoBERTa (OpenAI GPT-2) 0.79.

The AUROC gains don't reach the operating point: no combination catches more AI text at the 1% target than its base
by more than noise, and the two largest slim gains lose red-team samples. All stay readings, off the score.
