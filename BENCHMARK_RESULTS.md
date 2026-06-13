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

## Rejected Stage 4 Item Experiments

These runs are useful negative results. They were not promoted because they did
not improve held-out winrate over `item_bc_mlp_policy.json`.

### Counterfactual Value Classifier

Training:

```bash
python generate_item_value_dataset.py --samples 100000 --seed-start 8400000 --out datasets/item_value_100k_current_policy.npz --processes 0 --rollout-policy current
python train_item_model.py --dataset datasets/item_value_100k_current_policy.npz --epochs 100 --batch-size 16384 --hidden-sizes 768,512,256 --lr 0.0005 --weight-decay 0.00001 --val-split 0.1 --device cuda --out item_value_current_100k_policy.json
```

Benchmark:

```bash
python bench_flee_stage1.py --games 40000 --processes 0 --policies "combined:flee_ppo_policy.json,replay_ppo_policy.json,break_bc_mlp_policy.json,item_bc_mlp_policy.json" "combined:flee_ppo_policy.json,replay_ppo_policy.json,break_bc_mlp_policy.json,item_value_current_100k_policy.json"
```

| Policy | Played seats | Win% | Death% | Flee% | Clear% | Draw% | Breaks/game | Item use% | AvgScore | MedianScore |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| promoted combined learned | 46,683 | 30.80 +/- 0.42 | 32.31 | 61.28 | 6.70 | 28.93 | 0.228 | 56.59 | 5.491 | 5 |
| counterfactual item candidate | 46,639 | 28.84 +/- 0.41 | 34.15 | 60.05 | 6.19 | 28.99 | 0.242 | 38.97 | 5.202 | 5 |

Item hooks:

- promoted combined: `en_combat=328004/585144 (56.1%)`,
  `en_survie=7229/7229 (100.0%)`
- counterfactual item candidate: `en_combat=229950/601567 (38.2%)`,
  `en_survie=7351/7351 (100.0%)`

Diagnosis: single-rollout terminal-value labels were too conservative for combat
items.

### PPO Item Activation

Training:

```bash
python train_item_ppo.py --timesteps 0 --seed 9100000 --out item_ppo_bc1m_5ep.zip --rollout-policy current --bc-dataset datasets/item_activation_v2_1m.npz --bc-epochs 5 --bc-batch-size 32768 --bc-lr 0.0005 --device cuda --net-arch 512,512,256 --n-steps 1024 --batch-size 512 --learning-rate 0.0001
python train_item_ppo.py --load item_ppo_bc1m_5ep.zip --timesteps 100000 --seed 9200000 --out item_ppo_bc1m_100k.zip --rollout-policy current --device cuda
python export_item_ppo.py --model item_ppo_bc1m_100k.zip --out item_ppo_bc1m_100k_policy.json
```

Benchmark:

```bash
python bench_flee_stage1.py --games 40000 --processes 0 --policies "combined:flee_ppo_policy.json,replay_ppo_policy.json,break_bc_mlp_policy.json,item_bc_mlp_policy.json" "combined:flee_ppo_policy.json,replay_ppo_policy.json,break_bc_mlp_policy.json,item_ppo_bc1m_100k_policy.json"
```

| Policy | Played seats | Win% | Death% | Flee% | Clear% | Draw% | Breaks/game | Item use% | AvgScore | MedianScore |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| promoted combined learned | 69,976 | 29.02 +/- 0.34 | 36.23 | 55.96 | 8.08 | 29.16 | 0.211 | 57.67 | 5.212 | 5 |
| PPO item candidate | 70,005 | 27.43 +/- 0.33 | 36.79 | 55.58 | 7.93 | 29.24 | 0.196 | 64.11 | 5.029 | 4 |

Item hooks:

- promoted combined: `en_combat=489556/857500 (57.1%)`,
  `en_survie=11672/11672 (100.0%)`
- PPO item candidate: `en_combat=499438/785454 (63.6%)`,
  `en_survie=11437/11437 (100.0%)`

Diagnosis: the first PPO fine-tune pushed combat item activation too high and
lost winrate.

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
