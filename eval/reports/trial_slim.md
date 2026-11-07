# Trial combiner against the shipped one

Written by `python -m eval.trial_compare`. Red-team texts were in neither fit. A better trial catches more AI without flagging more human text.

| Texts | AI | Human | Combiner | AI caught | AI called human | Human flagged | Human called human | AUROC |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| daigt | 223 | 188 | shipped | 100% | 0% | 1% | 78% | 1.000 |
| daigt | 223 | 188 | trial | 100% | 0% | 1% | 91% | 1.000 |
| mage | 177 | 155 | shipped | 93% | 0% | 4% | 67% | 0.989 |
| mage | 177 | 155 | trial | 90% | 1% | 3% | 85% | 0.989 |
| raid | 131 | 102 | shipped | 86% | 2% | 3% | 89% | 0.986 |
| raid | 131 | 102 | trial | 86% | 3% | 0% | 92% | 0.986 |
| redteam_features, guided | 159 | 0 | shipped | 28% | 50% | - | - | - |
| redteam_features, guided | 159 | 0 | trial | 25% | 60% | - | - | - |
| redteam_features, public humanizers | 137 | 0 | shipped | 89% | 1% | - | - | - |
| redteam_features, public humanizers | 137 | 0 | trial | 87% | 4% | - | - | - |
| redteam_features, not guided | 389 | 3 | shipped | 51% | 16% | 0% | 100% | 0.929 |
| redteam_features, not guided | 389 | 3 | trial | 38% | 28% | 0% | 100% | 0.932 |
