# AlphaJon Benchmark Results

Benchmarks use the corrected SimuDonjon `ev` wrapper, including native
`worthit(...)` item-use behavior for combat items.

## Final Pairwise Benchmark

Command:

```bash
python bench_flee_stage1.py --games 80000 --processes 0 --policies ev "combined:flee_ppo_policy.json,replay_ppo_policy.json,break_bc_mlp_policy.json,item_bc_mlp_policy.json"
```

| Policy | Played seats | Win% | Death% | Flee% | Clear% | Draw% | Breaks/game | Item use% | AvgScore | MedianScore |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| corrected SimuDonjon `ev` | 139,948 | 25.22 +/- 0.23 | 49.39 | 42.11 | 8.84 | 2.21 | 0.166 | 58.42 | 4.307 | 2 |
| promoted combined learned | 139,941 | 30.93 +/- 0.24 | 30.99 | 63.42 | 5.93 | 29.17 | 0.237 | 56.03 | 5.526 | 5 |

Result: promoted combined learned policy is **+5.71 win-rate points** above the
corrected SimuDonjon `ev` baseline in this pairwise benchmark.

Item hooks:

- `ev`: `en_combat=965081/1671365 (57.7%)`, `en_survie=27315/27315 (100.0%)`
- promoted combined: `en_combat=981401/1768028 (55.5%)`,
  `en_survie=20888/20888 (100.0%)`

## Pairwise Head Ablations

Each row below is from a pairwise benchmark against corrected `ev`.

| Learned policy | Learned Win% | Baseline Win% | Delta | Notes |
| --- | ---: | ---: | ---: | --- |
| `fastppo:flee_ppo_policy.json` | 28.24 | 27.58 | +0.66 | Lower death, higher flee. |
| `replaymodel:replay_ppo_policy.json` | 31.73 | 23.89 | +7.84 | Strongest individual head. |
| `breakmodel:break_bc_mlp_policy.json` | 27.78 | 27.56 | +0.22 | Near-neutral alone. |
| `itemmodel:item_bc_mlp_policy.json` | 27.49 | 27.82 | -0.33 | Slightly lower alone, useful in full stack for aligned item behavior. |

## League Benchmark

The league benchmark includes all policies in the same mixed tables, so it is
not a clean pairwise SimuDonjon comparison. It is useful for checking behavior
against a broader policy pool.

Command:

```bash
python bench_flee_stage1.py --games 80000 --processes 0 --policies ev random fastppo:flee_ppo_policy.json replaymodel:replay_ppo_policy.json breakmodel:break_bc_mlp_policy.json itemmodel:item_bc_mlp_policy.json "combined:flee_ppo_policy.json,replay_ppo_policy.json,break_bc_mlp_policy.json,item_bc_mlp_policy.json"
```

| Policy | Win% | Death% | Flee% | AvgScore |
| --- | ---: | ---: | ---: | ---: |
| corrected SimuDonjon `ev` | 32.40 | 45.75 | 50.23 | 4.655 |
| random | 6.17 | 36.53 | 63.49 | 1.654 |
| flee only | 32.21 | 38.42 | 57.44 | 4.795 |
| replay only | 36.52 | 34.34 | 61.23 | 5.861 |
| break only | 26.59 | 48.05 | 44.75 | 4.422 |
| item only | 26.51 | 48.15 | 45.02 | 4.414 |
| promoted combined | 36.24 | 26.94 | 69.55 | 5.853 |
