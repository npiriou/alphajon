# AlphaJon

AlphaJon is a machine-learning fork of
[SimuDonjon](https://github.com/todiogame/simudonjon).

The original SimuDonjon project is a fast simulator for dungeon runs, item
drafts, and balance analysis. AlphaJon keeps that simulator as the rules engine
and adds policy interfaces, training environments, learned models, and
benchmarks so the game can be played by progressively learned AI agents.

## Current Goal

Train an AI that can play the full game while keeping the simulator trustworthy:

- decide when to flee or keep going;
- decide whether to replay/draw again;
- choose which item to break, discard, repair, or replace;
- decide when optional items should be used;
- eventually draft items and play full evenings with medals and hero levels.

The AI learns from structured game state, not from terminal logs or screen
pixels.

## Current Learning Stages

Implemented so far:

- Stage 1: flee / continue policy;
- Stage 2: replay / pass policy;
- Stage 3: object break policy;
- partial Stage 4: combat and survival item activation policy.

Planned next:

- finish full item-use coverage across all item hooks;
- draft picking;
- full evening strategy;
- self-play and checkpoint leagues.

See [LEARNING_PLAN.md](LEARNING_PLAN.md) for the detailed roadmap and current
coverage notes.

## Baseline Simulator Usage

The inherited SimuDonjon scripts are still available.

Run a normal dungeon simulation:

```bash
python donjon.py
```

Run a draft simulation:

```bash
python draft.py
```

Run full evening simulations:

```bash
python party.py
```

Pass a number to run detailed logged games:

```bash
python donjon.py 5
python draft.py 3
python party.py 2
```

## Learning And Benchmark Commands

Benchmark the current policies:

```bash
python bench_flee_stage1.py --games 20000 --policies ev model:flee_bc_mlp_policy.json
```

Benchmark the combined learned policy:

```bash
python bench_flee_stage1.py --games 20000 --policies ev "combined:flee_bc_mlp_policy.json,replay_ppo_policy.json,break_bc_mlp_policy.json,item_bc_mlp_policy.json"
```

Train supervised baselines:

```bash
python train_flee_model.py --samples 20000 --epochs 12 --out flee_bc_mlp_policy.json
python train_replay_model.py --samples 20000 --epochs 12 --out replay_bc_mlp_policy.json
python train_break_model.py --samples 20000 --epochs 12 --out break_bc_mlp_policy.json
python train_item_model.py --samples 20000 --epochs 12 --out item_bc_mlp_policy.json
```

Train PPO policies where available:

```bash
python train_flee_ppo.py --timesteps 50000 --out flee_ppo.zip
python train_replay_ppo.py --timesteps 50000 --out replay_ppo.zip
```

Export PPO actors to the lightweight JSON runtime format:

```bash
python export_flee_ppo.py --model flee_ppo.zip --out flee_ppo_policy.json
python export_flee_ppo.py --model replay_ppo.zip --out replay_ppo_policy.json
```

## Project Structure

- `simu.py`: main dungeon turn loop and rules execution.
- `donjon.py`, `draft.py`, `party.py`: inherited simulator entry points.
- `joueurs.py`: player state and legacy heuristic decisions.
- `objets.py`: item definitions and hook-based item effects.
- `heros.py`, `monstres.py`: hero and dungeon card definitions.
- `policies/`: policy interface, learned runtime policies, and feature extractors.
- `*_env.py`, `gym_*_env.py`: replayable training environments.
- `train_*_model.py`: supervised imitation trainers.
- `train_*_ppo.py`: Stable-Baselines3 PPO trainers.
- `bench_flee_stage1.py`: policy benchmark runner.
- `LEARNING_PLAN.md`: roadmap and coverage plan.

## Installation

Use Python 3.10+.

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dependencies

Core dependencies are listed in `requirements.txt`.

Current ML work uses:

- `numpy`;
- `torch`;
- `gymnasium`;
- `stable-baselines3`.

## Upstream

This repository is derived from SimuDonjon:

- upstream project: <https://github.com/todiogame/simudonjon>
- AlphaJon focus: learned decision policies and training infrastructure

Rules-engine changes should stay compatible with SimuDonjon behavior unless a
change is explicitly part of the learning roadmap.
