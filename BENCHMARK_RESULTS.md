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
| corrected SimuDonjon `ev` | 93,337 | 24.35 +/- 0.28 | 50.46 | 40.28 | 9.56 | 2.21 | 0.157 | 59.00 | 4.218 | 1 |
| promoted combined learned | 93,213 | 30.34 +/- 0.30 | 31.97 | 61.43 | 6.91 | 29.13 | 0.230 | 52.99 | 5.491 | 5 |

Result: promoted combined learned policy is **+5.99 win-rate points** above the
corrected SimuDonjon `ev` baseline in this pairwise benchmark.

Item hooks:

- `ev`: `en_combat=642223/1101456 (58.3%)`, `en_survie=18651/18651 (100.0%)`
- promoted combined: `en_combat=615044/1173241 (52.4%)`,
  `en_survie=14126/14126 (100.0%)`

Current promoted item head:

- `item_bc_mlp_policy.json` is now a build-aware pairwise Q model:
  `item_activation_q_tanh`, feature version `item_activation_v3`.
- Previous v2 imitation item head is saved as
  `item_bc_mlp_policy_v2_promoted_before_q.json`.

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

### Build-Aware Item Q

This run addresses the main representational flaw in the earlier item models:
`item_activation_v2` knew the current item but did not properly encode the full
build. `item_activation_v3` adds own item class bags by intact/broken/active
state, visible opponent item class bags, active opponent item bags, and build
summary scalars.

Training:

```bash
python generate_item_value_dataset.py --samples 500000 --seed-start 13300000 --out datasets/item_value_v3_500k_current.npz --processes 0 --rollout-policy current --forced-item-rounds 4 --win-weight 8.0 --death-weight 2.0 --score-weight 0.05
python train_item_q_model.py --dataset datasets/item_value_v3_500k_current.npz --epochs 100 --batch-size 65536 --hidden-sizes 768,512,256 --lr 0.0004 --weight-decay 0.00001 --device cuda --out item_q_v3_500k_compact_policy.json
```

Benchmark:

```bash
python bench_flee_stage1.py --games 80000 --processes 0 --policies ev "combined:flee_ppo_policy.json,replay_ppo_policy.json,break_bc_mlp_policy.json,item_bc_mlp_policy.json" "combined:flee_ppo_policy.json,replay_ppo_policy.json,break_bc_mlp_policy.json,item_q_v3_500k_compact_policy.json"
```

| Policy | Played seats | Win% | Death% | Flee% | Clear% | Draw% | Breaks/game | Item use% | AvgScore | MedianScore |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| corrected SimuDonjon `ev` | 93,337 | 24.38 +/- 0.28 | 50.52 | 40.26 | 9.52 | 2.19 | 0.156 | 58.95 | 4.209 | 1 |
| promoted combined learned | 93,339 | 30.04 +/- 0.29 | 32.68 | 60.64 | 6.97 | 29.14 | 0.230 | 56.62 | 5.448 | 5 |
| build-aware Q item candidate | 93,213 | 30.08 +/- 0.29 | 32.27 | 61.13 | 6.92 | 29.08 | 0.229 | 54.01 | 5.450 | 5 |

Item hooks:

- promoted combined: `en_combat=656055/1169844 (56.1%)`,
  `en_survie=14613/14613 (100.0%)`
- build-aware Q item candidate: `en_combat=630741/1179999 (53.5%)`,
  `en_survie=14213/14213 (100.0%)`

Diagnosis: build-aware Q is the first item value candidate that does not regress
the full stack, but its winrate gain is too small to treat as a decisive
promotion. The direction is correct; the remaining blocker is label quality and
per-item variance, not just network size.

### Build-Aware Pairwise Item Q

The value-only Q model still overfit noisy absolute value targets. The next
candidate trained the same Q architecture with an added pairwise/ranking loss on
`Q(use) - Q(skip)`, selecting checkpoints by validation regret.

Training:

```bash
python train_item_q_pairwise.py --dataset datasets/item_value_v3_500k_current.npz --epochs 100 --batch-size 65536 --hidden-sizes 768,512,256 --lr 0.0004 --weight-decay 0.00001 --rank-weight 0.75 --device cuda --out item_q_v3_500k_pairwise_policy.json
```

Confirmation benchmark before promotion:

```bash
python bench_flee_stage1.py --games 80000 --processes 0 --policies ev "combined:flee_ppo_policy.json,replay_ppo_policy.json,break_bc_mlp_policy.json,item_bc_mlp_policy.json" "combined:flee_ppo_policy.json,replay_ppo_policy.json,break_bc_mlp_policy.json,item_q_v3_500k_pairwise_policy.json"
```

| Policy | Played seats | Win% | Death% | Flee% | Clear% | Draw% | Breaks/game | Item use% | AvgScore | MedianScore |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| corrected SimuDonjon `ev` | 93,337 | 24.35 +/- 0.28 | 50.46 | 40.28 | 9.56 | 2.21 | 0.157 | 59.00 | 4.218 | 1 |
| previous promoted combined | 93,339 | 29.87 +/- 0.29 | 32.83 | 60.53 | 6.93 | 29.13 | 0.229 | 56.63 | 5.441 | 5 |
| pairwise Q item candidate | 93,213 | 30.34 +/- 0.30 | 31.97 | 61.43 | 6.91 | 29.13 | 0.230 | 52.99 | 5.491 | 5 |

Item hooks:

- previous promoted combined: `en_combat=656703/1170763 (56.1%)`,
  `en_survie=14621/14621 (100.0%)`
- pairwise Q item candidate: `en_combat=615044/1173241 (52.4%)`,
  `en_survie=14126/14126 (100.0%)`

Result: promoted. This is the first item-value head that clearly improves
full-stack winrate and death rate over the v2 imitation item model.

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
