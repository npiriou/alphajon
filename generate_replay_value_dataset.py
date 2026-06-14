import argparse
import json
import multiprocessing
import random
from pathlib import Path

import numpy as np

from policies import CombinedPolicy, HeuristicPolicy, NumpyBreakPolicy, NumpyItemActivationPolicy, NumpyPPOFleePolicy, NumpyReplayPolicy
from policies.replay_features import observation_size
from replay_env import ReplayEnv
from train_replay_model import state_from_replay


def build_rollout_policy(mode):
    if mode == "heuristic":
        return HeuristicPolicy("ev")
    flee = NumpyPPOFleePolicy("flee_ppo_policy.json")
    replay = NumpyReplayPolicy("replay_ppo_policy.json", flee_policy=flee)
    break_policy = NumpyBreakPolicy("break_bc_mlp_policy.json", flee_policy=flee, replay_policy=replay)
    item = NumpyItemActivationPolicy(
        "item_bc_mlp_policy.json",
        flee_policy=flee,
        replay_policy=replay,
        break_policy=break_policy,
    )
    return CombinedPolicy(flee_policy=flee, replay_policy=replay, break_policy=break_policy, item_policy=item)


def terminal_value(info, reward, win_weight, death_weight, score_weight):
    value = float(reward)
    if info["win"]:
        value += float(win_weight)
    if info["death"]:
        value -= float(death_weight)
    value += float(score_weight) * float(info["score"])
    return value


def evaluate_action(seed, prefix, action, rollout_policy, win_weight, death_weight, score_weight):
    env = ReplayEnv()
    _, reward, info = env.run_to_terminal(seed, list(prefix) + [int(action)], rollout_policy=rollout_policy)
    value = terminal_value(info, reward, win_weight, death_weight, score_weight)
    return value, reward, info


def collect_dataset(samples, seed_start, rollout_mode, win_weight, death_weight, score_weight, report_every):
    rollout_policy = build_rollout_policy(rollout_mode)
    xs = []
    seeds = []
    decision_indices = []
    value0 = []
    value1 = []
    reward0 = []
    reward1 = []
    win0 = []
    win1 = []
    death0 = []
    death1 = []
    seed = int(seed_start)
    env_count = 0

    while len(xs) < samples:
        env = ReplayEnv()
        env.rollout_policy = rollout_policy
        obs, _ = env.reset(seed=seed)
        env_count += 1
        seed += 1
        if env.terminal_players is not None:
            continue

        done = False
        while not done and len(xs) < samples:
            try:
                state_from_replay(env)
            except RuntimeError:
                break
            prefix = list(env._actions)
            v0, r0, i0 = evaluate_action(env.seed_value, prefix, 0, rollout_policy, win_weight, death_weight, score_weight)
            v1, r1, i1 = evaluate_action(env.seed_value, prefix, 1, rollout_policy, win_weight, death_weight, score_weight)
            label = 1 if v1 > v0 else 0
            xs.append(obs.copy())
            seeds.append(env.seed_value)
            decision_indices.append(len(prefix))
            value0.append(v0)
            value1.append(v1)
            reward0.append(r0)
            reward1.append(r1)
            win0.append(bool(i0["win"]))
            win1.append(bool(i1["win"]))
            death0.append(bool(i0["death"]))
            death1.append(bool(i1["death"]))
            obs, _, done, _, _ = env.step(label)

            if report_every and len(xs) % report_every == 0:
                print(
                    f"collected={len(xs)} envs={env_count} positive={float(np.mean(np.asarray(value1) > np.asarray(value0))):.3f} "
                    f"mean_delta={float(np.mean(np.asarray(value1) - np.asarray(value0))):.4f}"
                )

    arrays = {
        "x": np.vstack(xs).astype(np.float32),
        "seed": np.asarray(seeds, dtype=np.int64),
        "decision_index": np.asarray(decision_indices, dtype=np.int32),
        "value0": np.asarray(value0, dtype=np.float32),
        "value1": np.asarray(value1, dtype=np.float32),
        "reward0": np.asarray(reward0, dtype=np.float32),
        "reward1": np.asarray(reward1, dtype=np.float32),
        "win0": np.asarray(win0, dtype=bool),
        "win1": np.asarray(win1, dtype=bool),
        "death0": np.asarray(death0, dtype=bool),
        "death1": np.asarray(death1, dtype=bool),
        "y": (np.asarray(value1) > np.asarray(value0)).astype(np.int64),
    }
    metadata = {
        "label_source": "counterfactual_terminal_value",
        "decision": "replay",
        "rollout_policy": rollout_mode,
        "samples": int(len(xs)),
        "observation_size": observation_size(),
        "seed_start": int(seed_start),
        "seed_end_exclusive": int(seed),
        "envs": int(env_count),
        "positive_rate": float(np.mean(arrays["y"])) if len(xs) else 0.0,
        "win_weight": float(win_weight),
        "death_weight": float(death_weight),
        "score_weight": float(score_weight),
    }
    return arrays, metadata


def write_outputs(out, arrays, metadata):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    meta_path = out.with_suffix(out.suffix + ".json")
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)
    print(f"wrote {out}: samples={metadata['samples']} positive={metadata['positive_rate']:.3f}")
    print(f"wrote {meta_path}")


def _collect_worker(args):
    return collect_dataset(*args)


def collect_dataset_parallel(samples, seed_start, rollout_mode, win_weight, death_weight, score_weight, report_every, processes):
    if processes <= 1:
        return collect_dataset(samples, seed_start, rollout_mode, win_weight, death_weight, score_weight, report_every)
    base, rest = divmod(samples, processes)
    jobs = []
    seed = seed_start
    for idx in range(processes):
        count = base + (1 if idx < rest else 0)
        if count <= 0:
            continue
        jobs.append((count, seed, rollout_mode, win_weight, death_weight, score_weight, report_every if idx == 0 else 0))
        seed += 100000000
    with multiprocessing.Pool(processes) as pool:
        results = pool.map(_collect_worker, jobs)
    keys = ("x", "seed", "decision_index", "value0", "value1", "reward0", "reward1", "win0", "win1", "death0", "death1", "y")
    arrays = {key: np.concatenate([result[0][key] for result in results], axis=0) for key in keys}
    metadata = {
        "label_source": "counterfactual_terminal_value",
        "decision": "replay",
        "rollout_policy": rollout_mode,
        "samples": int(len(arrays["y"])),
        "observation_size": observation_size(),
        "seed_start": int(seed_start),
        "seed_end_exclusive": int(max(result[1]["seed_end_exclusive"] for result in results)),
        "envs": int(sum(result[1]["envs"] for result in results)),
        "positive_rate": float(np.mean(arrays["y"])) if len(arrays["y"]) else 0.0,
        "win_weight": float(win_weight),
        "death_weight": float(death_weight),
        "score_weight": float(score_weight),
        "processes": int(processes),
    }
    return arrays, metadata


def main():
    parser = argparse.ArgumentParser(description="Generate replay/pass Q-value data.")
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--seed-start", type=int, default=11000000)
    parser.add_argument("--out", default="datasets/replay_value_100k_current.npz")
    parser.add_argument("--rollout-policy", default="current", choices=("current", "heuristic"))
    parser.add_argument("--win-weight", type=float, default=8.0)
    parser.add_argument("--death-weight", type=float, default=2.0)
    parser.add_argument("--score-weight", type=float, default=0.05)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--report-every", type=int, default=10000)
    args = parser.parse_args()

    processes = args.processes
    if processes == 0:
        processes = max(1, (multiprocessing.cpu_count() or 2) - 1)
    random.seed(args.seed_start)
    np.random.seed(args.seed_start & 0xFFFFFFFF)
    arrays, metadata = collect_dataset_parallel(
        args.samples,
        args.seed_start,
        args.rollout_policy,
        args.win_weight,
        args.death_weight,
        args.score_weight,
        args.report_every,
        processes,
    )
    write_outputs(args.out, arrays, metadata)


if __name__ == "__main__":
    main()
