# SID 141390 fixed-tb score sensitivity

Scoring: `case=floor(100*tb/(tb+tk))`; leaderboard score is `sum(case)/12-10`. SID 141408 case-3 `tb` is excluded.

| c9/c10 `tk` change | c9/c10 case score | integer sum | leaderboard score | gain | gap to 75 |
|---:|---:|---:|---:|---:|---:|
| 0% | 75 / 76 | 972 | 71.000 | 0.000 | 4.000 |
| -10% | 77 / 78 | 976 | 71.333 | 0.333 | 3.667 |
| -20% | 79 / 80 | 980 | 71.667 | 0.667 | 3.333 |
| -25% | 80 / 81 | 982 | 71.833 | 0.833 | 3.167 |
| -30% | 81 / 82 | 984 | 72.000 | 1.000 | 3.000 |
| -40% | 84 / 84 | 989 | 72.417 | 1.417 | 2.583 |

At `tk -20%`, per-case integer gains c1..c12 are `+3,+3,+2,+2,+3,+3,+3,+3,+4,+4,+3,+3`.
