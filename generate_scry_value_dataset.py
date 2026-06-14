import argparse
import json
import random
from collections import Counter

import numpy as np

from policies import CombinedPolicy, HeuristicPolicy, NumpyBreakPolicy, NumpyItemActivationPolicy, NumpyPPOFleePolicy, NumpyReplayPolicy
from scry_env import ScryDecisionEnv


def build_rollout_policy(mode):
    if mode == "heuristic":
        return HeuristicPolicy("ev")
    flee = NumpyPPOFleePolicy("flee_ppo_policy.json")
    replay = NumpyReplayPolicy("replay_ppo_policy.json", flee_policy=flee)
    break_policy = NumpyBreakPolicy("break_bc_mlp_policy.json", flee_policy=flee, replay_policy=replay)
    item = NumpyItemActivationPolicy("item_bc_mlp_policy.json", flee_policy=flee, replay_policy=replay, break_policy=break_policy)
    return CombinedPolicy(flee_policy=flee, replay_policy=replay, break_policy=break_policy, item_policy=item)


def terminal_value(info, reward, win_weight, death_weight, score_weight):
    value = float(reward)
    if info["win"]:
        value += float(win_weight)
    if info["death"]:
        value -= float(death_weight)
    value += float(score_weight) * float(info["score"])
    return value


def evaluate_action(seed, prefix, action, rollout_policy, win_weight, death_weight, score_weight, controlled_initial_pv):
    env = ScryDecisionEnv(controlled_initial_pv=controlled_initial_pv)
    _, reward, info = env.run_to_terminal(seed, list(prefix) + [int(action)], rollout_policy=rollout_policy)
    return terminal_value(info, reward, win_weight, death_weight, score_weight), reward, info


def collect(samples, seed_start, rollout_mode, win_weight, death_weight, score_weight, controlled_initial_pv, report_every):
    rollout_policy = build_rollout_policy(rollout_mode)
    xs = []
    seeds = []
    decision_indices = []
    sources = []
    values = []
    best_actions = []
    baseline_actions = []
    coverage = Counter()
    seed = int(seed_start)
    env_count = 0
    while len(xs) < samples:
        env = ScryDecisionEnv(controlled_initial_pv=controlled_initial_pv)
        env.rollout_policy = rollout_policy
        obs, info = env.reset(seed=seed)
        env_count += 1
        seed += 1
        if env.terminal_players is not None:
            continue
        done = False
        while not done and len(xs) < samples:
            prefix = list(env._actions)
            legal = info.get("legal_actions", [0])
            action_values = np.full(4, -1.0e9, dtype=np.float32)
            for action in legal:
                value, _, _ = evaluate_action(
                    env.seed_value, prefix, action, rollout_policy, win_weight, death_weight, score_weight, controlled_initial_pv
                )
                action_values[int(action)] = value
            best = int(np.argmax(action_values))
            xs.append(obs.copy())
            seeds.append(env.seed_value)
            decision_indices.append(len(prefix))
            sources.append(info.get("source", ""))
            values.append(action_values)
            best_actions.append(best)
            baseline_actions.append(int(info.get("baseline_action", -1)))
            coverage[(info.get("source", ""), best)] += 1
            obs, _, done, _, info = env.step(best)
            if report_every and len(xs) % report_every == 0:
                print(f"collected={len(xs)} envs={env_count} best={dict(Counter(best_actions))}")
    arrays = {
        "x": np.vstack(xs).astype(np.float32),
        "seed": np.asarray(seeds, dtype=np.int64),
        "decision_index": np.asarray(decision_indices, dtype=np.int32),
        "source": np.asarray(sources),
        "values": np.vstack(values).astype(np.float32),
        "best_action": np.asarray(best_actions, dtype=np.int64),
        "baseline_action": np.asarray(baseline_actions, dtype=np.int64),
    }
    metadata = {
        "label_source": "scry_counterfactual_terminal_value",
        "rollout_policy": rollout_mode,
        "samples": len(xs),
        "seed_start": int(seed_start),
        "seed_end_exclusive": int(seed),
        "envs": int(env_count),
        "win_weight": float(win_weight),
        "death_weight": float(death_weight),
        "score_weight": float(score_weight),
        "controlled_initial_pv": controlled_initial_pv,
        "best_action_counts": dict(Counter(int(a) for a in best_actions)),
        "coverage": [
            {"source": source, "best_action": int(action), "count": int(count)}
            for (source, action), count in sorted(coverage.items())
        ],
    }
    return arrays, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed-start", type=int, default=20000000)
    parser.add_argument("--rollout-policy", choices=("heuristic", "current"), default="current")
    parser.add_argument("--win-weight", type=float, default=8.0)
    parser.add_argument("--death-weight", type=float, default=2.0)
    parser.add_argument("--score-weight", type=float, default=0.05)
    parser.add_argument("--controlled-initial-pv", type=int, default=None)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    random.seed(args.seed_start)
    np.random.seed(args.seed_start & 0xFFFFFFFF)
    arrays, metadata = collect(
        args.samples,
        args.seed_start,
        args.rollout_policy,
        args.win_weight,
        args.death_weight,
        args.score_weight,
        args.controlled_initial_pv,
        args.report_every,
    )
    np.savez_compressed(args.out, **arrays)
    with open(args.out + ".json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"wrote {args.out}: samples={metadata['samples']} envs={metadata['envs']}")


if __name__ == "__main__":
    main()
