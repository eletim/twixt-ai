# Paired 4-ply random opening evaluation

Each pair uses identical opening moves and position hash with colors exchanged. W-D-L is from the left candidate's perspective. Paired result is better/equal/worse pairs, followed by candidate game-point rate.

| Candidate | vs Frozen paired result | Overall W-D-L | Red W-D-L | Black W-D-L | Draw | Avg moves |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| N0 | 12/112/76 (34.2%) | 132-10-258 | 98-8-94 | 34-2-164 | 10 (2.5%) | 20.5 |
| N2 | 84/108/8 (68.6%) | 270-9-121 | 173-3-24 | 97-6-97 | 9 (2.2%) | 18.3 |
| N4 | 69/124/7 (65.4%) | 259-5-136 | 174-1-25 | 85-4-111 | 5 (1.2%) | 17.7 |
| N6 | 70/121/9 (64.5%) | 255-6-139 | 169-0-31 | 86-6-108 | 6 (1.5%) | 17.6 |

| Candidate pair | Better / Equal / Worse | Overall W-D-L | Both win / Split / Both loss | Opening terminal |
| --- | ---: | ---: | ---: | ---: |
| n0-vs-n2 | 0/29/71 (15.5%) | 29-4-167 | 0/29/67 | 0 |
| n0-vs-n4 | 3/34/63 (19.8%) | 38-3-159 | 1/34/62 | 0 |
| n0-vs-n6 | 2/44/54 (24.2%) | 47-3-150 | 1/44/52 | 0 |
| n2-vs-n4 | 25/68/7 (58.2%) | 114-5-81 | 21/68/6 | 0 |
| n2-vs-n6 | 17/73/10 (53.2%) | 105-3-92 | 15/73/9 | 0 |
| n4-vs-n6 | 14/71/15 (50.2%) | 99-3-98 | 14/71/12 | 0 |

## Opening diagnostics

The 200-pair suite has 200 unique positions; maximum multiplicity 1. Side to move is Red after four plies. The 100-pair candidate comparisons use the exact first 100 positions of this suite.

| Matchup | Games | Opening terminal | Both win / Split / Both loss | Red win rate |
| --- | ---: | ---: | ---: | ---: |
| n0-vs-frozen | 400 | 0 | 8/112/70 | 65.5% |
| n2-vs-frozen | 400 | 0 | 78/108/5 | 67.5% |
| n4-vs-frozen | 400 | 0 | 67/123/6 | 71.2% |
| n6-vs-frozen | 400 | 0 | 64/121/9 | 69.2% |

## Decision

Random-opening training improved playing strength against Frozen Gen11 for N=2, 4, and 6 on this suite. N=2 has the highest Frozen score (68.6%) and the most favorable direct pair comparison against Frozen (84 better, 8 worse). N=0 falls below Frozen (34.2%) and loses decisively to every random-opening candidate.

N=2 is the preferred setting for the next training run: it beats N=4 by 25 better to 7 worse pairs and N=6 by 17 to 10. The N=2 versus N=6 margin is smaller, so the exact ordering deserves another suite before promotion. N=4 and N=6 are effectively tied here (14 better, 15 worse).

Four-ply evaluation reduces Red wins to 65.5–71.2% in the Frozen matchups, compared with 399/400 Red wins in the earlier normal-start screens. Red remains favored, so role-swapped pair results are the primary strength measure. No champion promotion was performed.
