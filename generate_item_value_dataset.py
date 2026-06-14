import argparse
import json
import multiprocessing
import random
from collections import Counter
from pathlib import Path

import numpy as np

from item_env import ItemActivationEnv
from league_policy import LeaguePolicySampler
from objets import objets_disponibles
from policies import (
    CombinedPolicy,
    HeuristicPolicy,
    NumpyBreakPolicy,
    NumpyItemActivationPolicy,
    NumpyPPOFleePolicy,
    NumpyReplayPolicy,
)
from policies.item_features import FEATURE_VERSION, observation_size
from train_item_model import state_from_replay


def item_classes(focus_items=None):
    focus = {item.strip() for item in (focus_items or []) if item.strip()}
    seen = set()
    classes = []
    for item in objets_disponibles:
        cls = type(item)
        if focus and cls.__name__ not in focus and getattr(item, "nom", "") not in focus:
            continue
        if cls not in seen:
            seen.add(cls)
            classes.append(cls)
    if focus and not classes:
        raise ValueError(f"no item classes matched focus list: {sorted(focus)}")
    return classes


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


def build_opponent_sampler(path):
    return LeaguePolicySampler.from_json(path) if path else None


def terminal_value(info, reward, win_weight, death_weight, score_weight):
    value = float(reward)
    if info["win"]:
        value += float(win_weight)
    if info["death"]:
        value -= float(death_weight)
    value += float(score_weight) * float(info["score"])
    return value


def evaluate_action(
    seed,
    prefix,
    action,
    forced_item_class,
    rollout_policy,
    win_weight,
    death_weight,
    score_weight,
    scry_items_per_player,
    scry_hero_probability,
    opponent_policy_sampler,
    rollouts_per_action=1,
):
    values = []
    rewards = []
    infos = []
    for rollout_idx in range(max(1, int(rollouts_per_action))):
        env = ItemActivationEnv(
            forced_item_class=forced_item_class,
            scry_items_per_player=scry_items_per_player,
            scry_hero_probability=scry_hero_probability,
            opponent_policy_sampler=opponent_policy_sampler,
        )
        rollout_seed = None
        if int(rollouts_per_action) > 1:
            rollout_seed = int(seed) + 1000003 * (rollout_idx + 1) + 9176 * int(action)
        _, reward, info = env.run_to_terminal(
            seed,
            list(prefix) + [int(action)],
            rollout_policy=rollout_policy,
            rollout_seed_after_actions=rollout_seed,
        )
        values.append(terminal_value(info, reward, win_weight, death_weight, score_weight))
        rewards.append(float(reward))
        infos.append(info)
    merged = dict(infos[-1])
    merged["win_rate"] = float(np.mean([1.0 if info["win"] else 0.0 for info in infos]))
    merged["death_rate"] = float(np.mean([1.0 if info["death"] else 0.0 for info in infos]))
    merged["win"] = merged["win_rate"] >= 0.5
    merged["death"] = merged["death_rate"] >= 0.5
    return float(np.mean(values)), float(np.mean(rewards)), merged


def collect_dataset(
    samples,
    seed_start,
    forced_item_rounds,
    rollout_mode,
    margin,
    win_weight,
    death_weight,
    score_weight,
    force_survival,
    focus_items,
    record_focus_only,
    scry_items_per_player,
    scry_hero_probability,
    opponent_league,
    rollouts_per_action,
    report_every,
):
    rollout_policy = build_rollout_policy(rollout_mode)
    opponent_policy_sampler = build_opponent_sampler(opponent_league)
    focus = {item.strip() for item in (focus_items or []) if item.strip()}
    xs = []
    ys = []
    seeds = []
    decision_indices = []
    item_names = []
    item_classes_seen = []
    hooks = []
    card_names = []
    value0 = []
    value1 = []
    reward0 = []
    reward1 = []
    win0 = []
    win1 = []
    death0 = []
    death1 = []
    coverage = Counter()
    positive_coverage = Counter()
    tied = 0
    classes = item_classes(focus_items)
    seed = int(seed_start)
    env_count = 0

    while len(xs) < samples:
        forced_cls = None
        if forced_item_rounds > 0:
            forced_cls = classes[(env_count // forced_item_rounds) % len(classes)]
        env = ItemActivationEnv(
            forced_item_class=forced_cls,
            scry_items_per_player=scry_items_per_player,
            scry_hero_probability=scry_hero_probability,
            opponent_policy_sampler=opponent_policy_sampler,
        )
        env.rollout_policy = rollout_policy
        obs, _ = env.reset(seed=seed)
        env_count += 1
        seed += 1
        if env.terminal_players is not None:
            continue

        done = False
        while not done and len(xs) < samples:
            try:
                state, legal_actions = state_from_replay(env)
            except RuntimeError:
                break
            if 0 not in legal_actions or 1 not in legal_actions:
                action = legal_actions[0] if legal_actions else 0
                obs, _, done, _, _ = env.step(action)
                continue

            item = state["item"]
            card = state["card"]
            hook = state.get("hook", "")
            item_name = getattr(item, "nom", type(item).__name__)
            item_class = type(item).__name__
            if record_focus_only and focus and item_class not in focus and item_name not in focus:
                action = rollout_policy.choose_item_activation(state, legal_actions)
                obs, _, done, _, _ = env.step(action)
                continue

            prefix = list(env._actions)
            v0, r0, i0 = evaluate_action(
                env.seed_value,
                prefix,
                0,
                forced_cls,
                rollout_policy,
                win_weight,
                death_weight,
                score_weight,
                scry_items_per_player,
                scry_hero_probability,
                opponent_policy_sampler,
                rollouts_per_action,
            )
            v1, r1, i1 = evaluate_action(
                env.seed_value,
                prefix,
                1,
                forced_cls,
                rollout_policy,
                win_weight,
                death_weight,
                score_weight,
                scry_items_per_player,
                scry_hero_probability,
                opponent_policy_sampler,
                rollouts_per_action,
            )
            if force_survival and hook == "en_survie":
                label = 1
            elif abs(v1 - v0) < margin:
                label = 1 if i1["win"] and not i0["win"] else 0
                tied += 1
            else:
                label = 1 if v1 > v0 else 0

            xs.append(obs.copy())
            ys.append(label)
            seeds.append(env.seed_value)
            decision_indices.append(len(prefix))
            item_names.append(item_name)
            item_classes_seen.append(item_class)
            hooks.append(hook)
            card_names.append(getattr(card, "titre", ""))
            value0.append(v0)
            value1.append(v1)
            reward0.append(r0)
            reward1.append(r1)
            win0.append(bool(i0["win"]))
            win1.append(bool(i1["win"]))
            death0.append(bool(i0["death"]))
            death1.append(bool(i1["death"]))
            coverage[(item_class, hook)] += 1
            if label == 1:
                positive_coverage[(item_class, hook)] += 1

            obs, _, done, _, _ = env.step(label)

            if report_every and len(xs) % report_every == 0:
                print(
                    f"collected={len(xs)} envs={env_count} unique_item_hooks={len(coverage)} "
                    f"positive={float(np.mean(ys)):.3f} mean_delta={float(np.mean(np.asarray(value1) - np.asarray(value0))):.4f}"
                )

    metadata = {
        "label_source": "counterfactual_terminal_value",
        "rollout_policy": rollout_mode,
        "samples": len(xs),
        "feature_version": FEATURE_VERSION,
        "observation_size": observation_size(),
        "seed_start": int(seed_start),
        "seed_end_exclusive": int(seed),
        "envs": int(env_count),
        "forced_item_rounds": int(forced_item_rounds),
        "positive_rate": float(np.mean(ys)) if ys else 0.0,
        "tie_rate": float(tied / max(1, len(ys))),
        "margin": float(margin),
        "win_weight": float(win_weight),
        "death_weight": float(death_weight),
        "score_weight": float(score_weight),
        "force_survival": bool(force_survival),
        "focus_items": list(focus_items or []),
        "record_focus_only": bool(record_focus_only),
        "scry_items_per_player": int(scry_items_per_player),
        "scry_hero_probability": float(scry_hero_probability),
        "opponent_league": opponent_league,
        "rollouts_per_action": int(rollouts_per_action),
        "coverage": [
            {
                "item_class": item_class,
                "hook": hook,
                "decisions": int(count),
                "activations": int(positive_coverage.get((item_class, hook), 0)),
            }
            for (item_class, hook), count in sorted(coverage.items())
        ],
    }
    arrays = {
        "x": np.vstack(xs).astype(np.float32),
        "y": np.asarray(ys, dtype=np.int64),
        "seed": np.asarray(seeds, dtype=np.int64),
        "decision_index": np.asarray(decision_indices, dtype=np.int32),
        "item": np.asarray(item_names, dtype=object),
        "item_class": np.asarray(item_classes_seen, dtype=object),
        "hook": np.asarray(hooks, dtype=object),
        "card": np.asarray(card_names, dtype=object),
        "value0": np.asarray(value0, dtype=np.float32),
        "value1": np.asarray(value1, dtype=np.float32),
        "reward0": np.asarray(reward0, dtype=np.float32),
        "reward1": np.asarray(reward1, dtype=np.float32),
        "win0": np.asarray(win0, dtype=bool),
        "win1": np.asarray(win1, dtype=bool),
        "death0": np.asarray(death0, dtype=bool),
        "death1": np.asarray(death1, dtype=bool),
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
    print(f"wrote {meta_path}: item_hook_pairs={len(metadata['coverage'])}")


def _collect_worker(args):
    return collect_dataset(*args)


def merge_datasets(
    results,
    seed_start,
    forced_item_rounds,
    rollout_mode,
    margin,
    win_weight,
    death_weight,
    score_weight,
    force_survival,
    focus_items,
    record_focus_only,
    scry_items_per_player,
    scry_hero_probability,
    opponent_league,
    rollouts_per_action,
):
    keys = (
        "x",
        "y",
        "seed",
        "decision_index",
        "item",
        "item_class",
        "hook",
        "card",
        "value0",
        "value1",
        "reward0",
        "reward1",
        "win0",
        "win1",
        "death0",
        "death1",
    )
    arrays = {key: np.concatenate([result[0][key] for result in results], axis=0) for key in keys}
    coverage = Counter()
    positive_coverage = Counter()
    envs = 0
    seed_end = seed_start
    tied = 0.0
    total = 0
    for _, metadata in results:
        envs += metadata["envs"]
        seed_end = max(seed_end, metadata["seed_end_exclusive"])
        tied += metadata["tie_rate"] * metadata["samples"]
        total += metadata["samples"]
        for row in metadata["coverage"]:
            key = (row["item_class"], row["hook"])
            coverage[key] += row["decisions"]
            positive_coverage[key] += row["activations"]

    y = arrays["y"]
    metadata = {
        "label_source": "counterfactual_terminal_value",
        "rollout_policy": rollout_mode,
        "samples": int(len(y)),
        "feature_version": FEATURE_VERSION,
        "observation_size": observation_size(),
        "seed_start": int(seed_start),
        "seed_end_exclusive": int(seed_end),
        "envs": int(envs),
        "forced_item_rounds": int(forced_item_rounds),
        "positive_rate": float(np.mean(y)) if len(y) else 0.0,
        "tie_rate": float(tied / max(1, total)),
        "margin": float(margin),
        "win_weight": float(win_weight),
        "death_weight": float(death_weight),
        "score_weight": float(score_weight),
        "force_survival": bool(force_survival),
        "focus_items": list(focus_items or []),
        "record_focus_only": bool(record_focus_only),
        "scry_items_per_player": int(scry_items_per_player),
        "scry_hero_probability": float(scry_hero_probability),
        "opponent_league": opponent_league,
        "rollouts_per_action": int(rollouts_per_action),
        "coverage": [
            {
                "item_class": item_class,
                "hook": hook,
                "decisions": int(count),
                "activations": int(positive_coverage.get((item_class, hook), 0)),
            }
            for (item_class, hook), count in sorted(coverage.items())
        ],
    }
    return arrays, metadata


def collect_dataset_parallel(
    samples,
    seed_start,
    forced_item_rounds,
    rollout_mode,
    margin,
    win_weight,
    death_weight,
    score_weight,
    force_survival,
    focus_items,
    record_focus_only,
    scry_items_per_player,
    scry_hero_probability,
    opponent_league,
    rollouts_per_action,
    report_every,
    processes,
):
    if processes <= 1:
        return collect_dataset(
            samples,
            seed_start,
            forced_item_rounds,
            rollout_mode,
            margin,
            win_weight,
            death_weight,
            score_weight,
            force_survival,
            focus_items,
            record_focus_only,
            scry_items_per_player,
            scry_hero_probability,
            opponent_league,
            rollouts_per_action,
            report_every,
        )
    base, rest = divmod(samples, processes)
    jobs = []
    seed = seed_start
    for idx in range(processes):
        count = base + (1 if idx < rest else 0)
        if count <= 0:
            continue
        jobs.append(
            (
                count,
                seed,
                forced_item_rounds,
                rollout_mode,
                margin,
                win_weight,
                death_weight,
                score_weight,
                force_survival,
                focus_items,
                record_focus_only,
                scry_items_per_player,
                scry_hero_probability,
                opponent_league,
                rollouts_per_action,
                report_every if idx == 0 else 0,
            )
        )
        seed += 100000000
    with multiprocessing.Pool(processes) as pool:
        results = pool.map(_collect_worker, jobs)
    return merge_datasets(
        results,
        seed_start,
        forced_item_rounds,
        rollout_mode,
        margin,
        win_weight,
        death_weight,
        score_weight,
        force_survival,
        focus_items,
        record_focus_only,
        scry_items_per_player,
        scry_hero_probability,
        opponent_league,
        rollouts_per_action,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate item activation labels from counterfactual terminal value.")
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--seed-start", type=int, default=8000000)
    parser.add_argument("--out", default="datasets/item_value_50k.npz")
    parser.add_argument("--forced-item-rounds", type=int, default=4)
    parser.add_argument("--rollout-policy", default="current", choices=("current", "heuristic"))
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--win-weight", type=float, default=8.0)
    parser.add_argument("--death-weight", type=float, default=2.0)
    parser.add_argument("--score-weight", type=float, default=0.05)
    parser.add_argument("--no-force-survival", action="store_true")
    parser.add_argument(
        "--focus-items",
        default="",
        help="Comma-separated Python class names or display names to force-cycle instead of the full item registry.",
    )
    parser.add_argument(
        "--record-focus-only",
        action="store_true",
        help="When --focus-items is set, advance non-focus item decisions with the rollout policy without recording counterfactual labels.",
    )
    parser.add_argument("--scry-items-per-player", type=int, default=0)
    parser.add_argument("--scry-hero-probability", type=float, default=0.0)
    parser.add_argument("--opponent-league", default=None)
    parser.add_argument("--rollouts-per-action", type=int, default=1)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--report-every", type=int, default=5000)
    args = parser.parse_args()

    processes = args.processes
    if processes == 0:
        processes = max(1, (multiprocessing.cpu_count() or 2) - 1)
    random.seed(args.seed_start)
    np.random.seed(args.seed_start & 0xFFFFFFFF)
    arrays, metadata = collect_dataset_parallel(
        samples=args.samples,
        seed_start=args.seed_start,
        forced_item_rounds=args.forced_item_rounds,
        rollout_mode=args.rollout_policy,
        margin=args.margin,
        win_weight=args.win_weight,
        death_weight=args.death_weight,
        score_weight=args.score_weight,
        force_survival=not args.no_force_survival,
        focus_items=[part.strip() for part in args.focus_items.split(",") if part.strip()],
        record_focus_only=args.record_focus_only,
        scry_items_per_player=args.scry_items_per_player,
        scry_hero_probability=args.scry_hero_probability,
        opponent_league=args.opponent_league,
        rollouts_per_action=args.rollouts_per_action,
        report_every=args.report_every,
        processes=processes,
    )
    metadata["processes"] = processes
    write_outputs(args.out, arrays, metadata)


if __name__ == "__main__":
    main()
