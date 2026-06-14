import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from generate_item_value_dataset import build_rollout_policy, item_classes, terminal_value
from joint_decision_env import JointDecisionEnv
from policies.joint_features import FEATURE_VERSION, observation_size

MAX_ACTIONS = 4


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
    force_controlled_prophete,
    controlled_initial_pv,
):
    env = JointDecisionEnv(
        forced_item_class=forced_item_class,
        scry_items_per_player=scry_items_per_player,
        scry_hero_probability=scry_hero_probability,
        force_controlled_prophete=force_controlled_prophete,
        controlled_initial_pv=controlled_initial_pv,
    )
    _, reward, info = env.run_to_terminal(seed, list(prefix) + [int(action)], rollout_policy=rollout_policy)
    return terminal_value(info, reward, win_weight, death_weight, score_weight), reward, info


def collect(
    samples,
    seed_start,
    forced_item_rounds,
    rollout_mode,
    win_weight,
    death_weight,
    score_weight,
    focus_items,
    record_focus_only,
    scry_items_per_player,
    scry_hero_probability,
    force_controlled_prophete,
    controlled_initial_pv,
    record_kinds,
    report_every,
):
    rollout_policy = build_rollout_policy(rollout_mode)
    focus = {item.strip() for item in (focus_items or []) if item.strip()}
    classes = item_classes(focus_items)
    xs = []
    values = []
    legal_masks = []
    best_actions = []
    baseline_actions = []
    seeds = []
    decision_indices = []
    kinds = []
    item_classes_seen = []
    hooks = []
    sources = []
    coverage = Counter()
    baseline_legal = 0
    seed = int(seed_start)
    env_count = 0

    while len(xs) < samples:
        forced_cls = None
        if forced_item_rounds > 0:
            forced_cls = classes[(env_count // forced_item_rounds) % len(classes)]
        env = JointDecisionEnv(
            forced_item_class=forced_cls,
            scry_items_per_player=scry_items_per_player,
            scry_hero_probability=scry_hero_probability,
            force_controlled_prophete=force_controlled_prophete,
            controlled_initial_pv=controlled_initial_pv,
        )
        env.rollout_policy = rollout_policy
        obs, info = env.reset(seed=seed)
        env_count += 1
        seed += 1
        if env.terminal_players is not None:
            continue

        done = False
        while not done and len(xs) < samples:
            kind = info.get("kind", "")
            if record_kinds and kind not in record_kinds:
                action = int(info.get("baseline_action", info.get("legal_actions", [0])[0]))
                obs, _, done, _, info = env.step(action)
                continue
            item_class = info.get("item_class", "")
            if record_focus_only and focus and kind == "item_activation":
                item_name = info.get("item", "")
                if item_class not in focus and item_name not in focus:
                    action = int(info.get("baseline_action", info.get("legal_actions", [0])[0]))
                    obs, _, done, _, info = env.step(action)
                    continue
            legal = [int(a) for a in info.get("legal_actions", []) if 0 <= int(a) < MAX_ACTIONS]
            if not legal:
                obs, _, done, _, info = env.step(0)
                continue
            prefix = list(env._actions)
            action_values = np.full(MAX_ACTIONS, -1.0e9, dtype=np.float32)
            for action in legal:
                value, _, _ = evaluate_action(
                    env.seed_value,
                    prefix,
                    action,
                    forced_cls,
                    rollout_policy,
                    win_weight,
                    death_weight,
                    score_weight,
                    scry_items_per_player,
                    scry_hero_probability,
                    force_controlled_prophete,
                    controlled_initial_pv,
                )
                action_values[action] = value
            best = int(np.argmax(action_values))
            baseline = int(info.get("baseline_action", legal[0]))
            xs.append(obs.copy())
            values.append(action_values)
            mask = np.zeros(MAX_ACTIONS, dtype=bool)
            mask[legal] = True
            legal_masks.append(mask)
            best_actions.append(best)
            baseline_actions.append(baseline)
            baseline_legal += int(baseline in legal)
            seeds.append(env.seed_value)
            decision_indices.append(len(prefix))
            kinds.append(kind)
            item_classes_seen.append(item_class)
            hooks.append(info.get("hook", ""))
            sources.append(info.get("source", ""))
            coverage[(kind, item_class, info.get("hook", ""), info.get("source", ""))] += 1
            obs, _, done, _, info = env.step(best)
            if report_every and len(xs) % report_every == 0:
                print(
                    f"collected={len(xs)} envs={env_count} kinds={dict(Counter(kinds))} "
                    f"best={dict(Counter(best_actions))}"
                )

    arrays = {
        "x": np.vstack(xs).astype(np.float32),
        "values": np.vstack(values).astype(np.float32),
        "legal_mask": np.vstack(legal_masks).astype(bool),
        "best_action": np.asarray(best_actions, dtype=np.int64),
        "baseline_action": np.asarray(baseline_actions, dtype=np.int64),
        "seed": np.asarray(seeds, dtype=np.int64),
        "decision_index": np.asarray(decision_indices, dtype=np.int32),
        "kind": np.asarray(kinds, dtype=object),
        "item_class": np.asarray(item_classes_seen, dtype=object),
        "hook": np.asarray(hooks, dtype=object),
        "source": np.asarray(sources, dtype=object),
    }
    metadata = {
        "label_source": "joint_counterfactual_terminal_value",
        "rollout_policy": rollout_mode,
        "feature_version": FEATURE_VERSION,
        "observation_size": observation_size(),
        "samples": int(len(xs)),
        "seed_start": int(seed_start),
        "seed_end_exclusive": int(seed),
        "envs": int(env_count),
        "forced_item_rounds": int(forced_item_rounds),
        "win_weight": float(win_weight),
        "death_weight": float(death_weight),
        "score_weight": float(score_weight),
        "focus_items": list(focus_items or []),
        "record_focus_only": bool(record_focus_only),
        "scry_items_per_player": int(scry_items_per_player),
        "scry_hero_probability": float(scry_hero_probability),
        "force_controlled_prophete": bool(force_controlled_prophete),
        "controlled_initial_pv": controlled_initial_pv,
        "record_kinds": sorted(record_kinds),
        "kind_counts": dict(Counter(kinds)),
        "best_action_counts": dict(Counter(int(a) for a in best_actions)),
        "baseline_legal_rate": float(baseline_legal / max(1, len(xs))),
        "coverage": [
            {
                "kind": kind,
                "item_class": item_class,
                "hook": hook,
                "source": source,
                "decisions": int(count),
            }
            for (kind, item_class, hook, source), count in sorted(coverage.items())
        ],
    }
    return arrays, metadata


def main():
    parser = argparse.ArgumentParser(description="Generate mixed item/scry Q(state, action) terminal-value labels.")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed-start", type=int, default=24000000)
    parser.add_argument("--out", required=True)
    parser.add_argument("--forced-item-rounds", type=int, default=4)
    parser.add_argument("--rollout-policy", choices=("current", "heuristic"), default="current")
    parser.add_argument("--win-weight", type=float, default=8.0)
    parser.add_argument("--death-weight", type=float, default=2.0)
    parser.add_argument("--score-weight", type=float, default=0.05)
    parser.add_argument("--focus-items", default="")
    parser.add_argument("--record-focus-only", action="store_true")
    parser.add_argument("--scry-items-per-player", type=int, default=2)
    parser.add_argument("--scry-hero-probability", type=float, default=0.75)
    parser.add_argument("--force-controlled-prophete", action="store_true")
    parser.add_argument("--controlled-initial-pv", type=int, default=None)
    parser.add_argument(
        "--record-kinds",
        default="",
        help="Comma-separated decision kinds to record, for example scry_window or item_activation.",
    )
    parser.add_argument("--report-every", type=int, default=100)
    args = parser.parse_args()
    random.seed(args.seed_start)
    np.random.seed(args.seed_start & 0xFFFFFFFF)
    focus_items = [part.strip() for part in args.focus_items.split(",") if part.strip()]
    record_kinds = {part.strip() for part in args.record_kinds.split(",") if part.strip()}
    arrays, metadata = collect(
        args.samples,
        args.seed_start,
        args.forced_item_rounds,
        args.rollout_policy,
        args.win_weight,
        args.death_weight,
        args.score_weight,
        focus_items,
        args.record_focus_only,
        args.scry_items_per_player,
        args.scry_hero_probability,
        args.force_controlled_prophete,
        args.controlled_initial_pv,
        record_kinds,
        args.report_every,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    with open(str(out) + ".json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)
    print(f"wrote {out}: samples={metadata['samples']} kinds={metadata['kind_counts']}")


if __name__ == "__main__":
    main()
