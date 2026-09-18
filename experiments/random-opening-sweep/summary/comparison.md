# Random opening sweep results

| N | Games | Training positions | Avg moves | Red win | Draw | Policy CE | Value MSE | Sign error | vs Frozen W-D-L |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 5000 | 65233 | 13.05 | 99.9% | 0.0% | 2.4699 | 0.0000 | 0.0% | 20-0-20 |
| 2 | 5000 | 77177 | 17.44 | 73.3% | 1.6% | 2.9662 | 0.4342 | 15.7% | 20-0-20 |
| 4 | 5000 | 72768 | 18.55 | 70.0% | 1.8% | 3.0214 | 0.3919 | 15.4% | 20-0-20 |
| 6 | 5000 | 69074 | 19.81 | 69.3% | 1.9% | 3.0098 | 0.3819 | 15.0% | 20-0-20 |

## Candidate pairwise

| Pair | First candidate W-D-L | Score |
| --- | ---: | ---: |
| n0-vs-n2 | 20-0-20 | 50.0% |
| n0-vs-n4 | 20-1-19 | 51.2% |
| n0-vs-n6 | 20-0-20 | 50.0% |
| n2-vs-n4 | 20-0-20 | 50.0% |
| n2-vs-n6 | 20-0-20 | 50.0% |
| n4-vs-n6 | 20-0-20 | 50.0% |

## Self-play and target distribution

| N | Median moves | Excluded positions | Positions/game | Red | Black | Draw | Games/hour | Sims/s | Effective batch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 13.0 | 0 | 13.05 | 4997 | 1 | 2 | 12365 | 2868 | 7.97 |
| 2 | 15.0 | 10000 | 15.44 | 3663 | 1255 | 82 | 9326 | 2559 | 7.97 |
| 4 | 16.0 | 20000 | 14.55 | 3501 | 1407 | 92 | 9825 | 2542 | 7.95 |
| 6 | 17.0 | 30000 | 13.81 | 3467 | 1438 | 95 | 10278 | 2524 | 7.95 |

Throughput was measured while some generation jobs ran concurrently; its differences cannot be attributed to N alone.

## Held-out Policy and Value diagnostics

| N | Policy entropy | Policy top1 | Policy top3 | Visit top1 agreement | Value MAE | Calibration error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2.4637 | 0.4876 | 0.6343 | 99.8% | 0.0002 | 0.0002 |
| 2 | 2.8558 | 0.3133 | 0.4941 | 56.7% | 0.3975 | 0.0941 |
| 4 | 2.9132 | 0.2942 | 0.4725 | 52.1% | 0.3772 | 0.0643 |
| 6 | 2.9295 | 0.2833 | 0.4617 | 52.0% | 0.3856 | 0.0437 |

Each row above uses that condition's own game-held-out validation set. The target distributions differ between rows.

## Common held-out comparison

Each model was also scored on every condition's validation set. Each cell is Policy CE / Value MSE.

| Test set | Frozen | N=0 | N=2 | N=4 | N=6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| N=0 | 2.481 / 0.000 | 2.470 / 0.000 | 2.532 / 0.011 | 2.545 / 0.009 | 2.537 / 0.008 |
| N=2 | 2.986 / 0.634 | 3.446 / 0.725 | 2.966 / 0.434 | 2.997 / 0.449 | 3.000 / 0.449 |
| N=4 | 3.001 / 0.580 | 3.458 / 0.674 | 3.008 / 0.395 | 3.021 / 0.392 | 3.017 / 0.383 |
| N=6 | 2.980 / 0.590 | 3.462 / 0.717 | 3.011 / 0.423 | 3.020 / 0.420 | 3.010 / 0.382 |

## Artifact hashes

| N | Dataset manifest SHA-256 | Best checkpoint SHA-256 |
| ---: | --- | --- |
| 0 | `7ecc95e400317edb102db44974fa20d2065f87c226683b5d1779a06f3e710bdc` | `4ed4944a0b0c4da1d34e15e0f1518aa73e29f0c993137386d12ec03f400cf8d5` |
| 2 | `d68525d574aa9c752b07f14b32239d2020049a59742b86cfd1e61f336701f2f4` | `6a44af92544e86801f1cc4981dfca99947a679df1b68264c66f314b14ed71e59` |
| 4 | `9dff2ff659606e80cb297124c0be0bdbeb803b610483f9183b4fc436c8577133` | `9cc5f2675d449e4ec925e2285894663953ebdc2e0301bbcacc4fab0d6d1ea86f` |
| 6 | `249a5b5bf503060254d7ab9c5d3645943947bb74296aae24891ad26085149cf3` | `08149bf57fcec31714779614d527f5f6fec7946db914c46a9d688d72252286ad` |

## Interpretation

Random opening reduced the extreme Red win rate and increased game length and training positions per game. N=2 produced the most training positions. N=6 had the lowest own-set Value MSE among opening conditions; N=2 had the lowest Policy CE among candidates on the N=2 and N=4 test sets.

Playing-strength improvement was not established. All four candidates scored 20/40 against Frozen Gen11. Five of six candidate pairs split 20-20; N=0 versus N=4 ended 20-1-19. Across all ten 40-game matchups, Red won 399 games and one game was drawn. The normal-start evaluation cannot identify a strongest N under this severe first-player effect. No promotion decision was made.
