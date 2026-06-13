import argparse
import json
import multiprocessing
import random
from collections import Counter
from pathlib import Path

import numpy as np

from item_env import ItemActivationEnv
from objets import objets_disponibles
from policies import HeuristicPolicy
from policies.item_features import FEATURE_VERSION, observation_size
from train_item_model import state_from_replay


def item_classes():
    seen = set()
    classes = []
    for item in objets_disponibles:
        cls = type(item)
        if cls not in seen:
            seen.add(cls)
            classes.append(cls)
    return classes


def baseline_label(state, legal_actions):
    return int(HeuristicPolicy("ev").choose_item_activation(state, legal_actions))


def collect_dataset(samples, seed_start, forced_item_rounds, report_every):
    xs = []
    ys = []
    seeds = []
    decision_indices = []
    item_names = []
    item_classes_seen = []
    hooks = []
    card_names = []
    coverage = Counter()
    positive_coverage = Counter()
    classes = item_classes()
    seed = int(seed_start)
    env_count = 0

    while len(xs) < samples:
        forced_cls = None
        if forced_item_rounds > 0:
            forced_cls = classes[(env_count // forced_item_rounds) % len(classes)]
        env = ItemActivationEnv(forced_item_class=forced_cls)
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
            action = baseline_label(state, legal_actions)
            item = state["item"]
            card = state["card"]
            hook = state.get("hook", "")
            item_name = getattr(item, "nom", type(item).__name__)
            item_class = type(item).__name__

            xs.append(obs.copy())
            ys.append(action)
            seeds.append(env.seed_value)
            decision_indices.append(len(env._actions))
            item_names.append(item_name)
            item_classes_seen.append(item_class)
            hooks.append(hook)
            card_names.append(getattr(card, "titre", ""))
            coverage[(item_class, hook)] += 1
            if action == 1:
                positive_coverage[(item_class, hook)] += 1

            obs, _, done, _, _ = env.step(action)

            if report_every and len(xs) % report_every == 0:
                print(
                    f"collected={len(xs)} envs={env_count} "
                    f"unique_item_hooks={len(coverage)} positive={float(np.mean(ys)):.3f}"
                )

    metadata = {
        "samples": len(xs),
        "feature_version": FEATURE_VERSION,
        "observation_size": observation_size(),
        "seed_start": int(seed_start),
        "seed_end_exclusive": seed,
        "envs": env_count,
        "forced_item_rounds": int(forced_item_rounds),
        "positive_rate": float(np.mean(ys)) if ys else 0.0,
        "coverage": [
            {
                "item_class": item_class,
                "hook": hook,
                "decisions": count,
                "activations": positive_coverage.get((item_class, hook), 0),
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
    samples, seed_start, forced_item_rounds, report_every = args
    return collect_dataset(samples, seed_start, forced_item_rounds, report_every)


def merge_datasets(results, seed_start, forced_item_rounds):
    arrays = {}
    for key in ("x", "y", "seed", "decision_index", "item", "item_class", "hook", "card"):
        arrays[key] = np.concatenate([result[0][key] for result in results], axis=0)

    coverage = Counter()
    positive_coverage = Counter()
    envs = 0
    seed_end = seed_start
    for _, metadata in results:
        envs += metadata["envs"]
        seed_end = max(seed_end, metadata["seed_end_exclusive"])
        for row in metadata["coverage"]:
            key = (row["item_class"], row["hook"])
            coverage[key] += row["decisions"]
            positive_coverage[key] += row["activations"]

    y = arrays["y"]
    metadata = {
        "samples": int(len(y)),
        "feature_version": FEATURE_VERSION,
        "observation_size": observation_size(),
        "seed_start": int(seed_start),
        "seed_end_exclusive": int(seed_end),
        "envs": int(envs),
        "forced_item_rounds": int(forced_item_rounds),
        "positive_rate": float(np.mean(y)) if len(y) else 0.0,
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


def collect_dataset_parallel(samples, seed_start, forced_item_rounds, report_every, processes):
    if processes <= 1:
        return collect_dataset(samples, seed_start, forced_item_rounds, report_every)
    base, rest = divmod(samples, processes)
    jobs = []
    seed = seed_start
    for idx in range(processes):
        count = base + (1 if idx < rest else 0)
        if count <= 0:
            continue
        jobs.append((count, seed, forced_item_rounds, report_every if idx == 0 else 0))
        seed += 100000000
    with multiprocessing.Pool(processes) as pool:
        results = pool.map(_collect_worker, jobs)
    return merge_datasets(results, seed_start, forced_item_rounds)


def main():
    parser = argparse.ArgumentParser(description="Generate reusable item activation imitation data.")
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--seed-start", type=int, default=2000000)
    parser.add_argument("--out", default="datasets/item_activation_100k.npz")
    parser.add_argument(
        "--forced-item-rounds",
        type=int,
        default=4,
        help="Number of generated games to keep each forced item before cycling. Use 0 for pure random.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Parallel rollout workers. Use 0 for cpu_count - 1.",
    )
    parser.add_argument("--report-every", type=int, default=10000)
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
        report_every=args.report_every,
        processes=processes,
    )
    metadata["processes"] = processes
    write_outputs(args.out, arrays, metadata)


if __name__ == "__main__":
    main()
