import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_q_policy(path):
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if payload.get("type") != "item_activation_q_tanh":
        raise ValueError(f"{path} is not an item_activation_q_tanh policy")
    layers = [
        (
            np.asarray(layer["weight"], dtype=np.float32),
            np.asarray(layer["bias"], dtype=np.float32),
        )
        for layer in payload["policy_layers"]
    ]
    q_weight = np.asarray(payload["q_weight"], dtype=np.float32)
    q_bias = np.asarray(payload["q_bias"], dtype=np.float32)
    q_use_bias = float(payload.get("metadata", {}).get("q_use_bias", 0.0))
    observation_size = int(payload["observation_size"])
    return layers, q_weight, q_bias, q_use_bias, observation_size


def predict_values(x, layers, q_weight, q_bias, q_use_bias, batch_size):
    skip = np.tile(np.asarray([1.0, 0.0], dtype=np.float32), (len(x), 1))
    use = np.tile(np.asarray([0.0, 1.0], dtype=np.float32), (len(x), 1))
    q0 = np.empty(len(x), dtype=np.float32)
    q1 = np.empty(len(x), dtype=np.float32)
    for start in range(0, len(x), batch_size):
        end = min(len(x), start + batch_size)
        for target, action_features, extra_bias in ((q0, skip[start:end], 0.0), (q1, use[start:end], q_use_bias)):
            h = np.concatenate([x[start:end], action_features], axis=1).astype(np.float32)
            for weight, bias in layers:
                h = np.tanh(h @ weight.T + bias)
            target[start:end] = (h @ q_weight.reshape(-1) + float(q_bias.reshape(-1)[0]) + extra_bias).astype(np.float32)
    return q0, q1


def summarize(dataset, policy, min_count, out):
    payload = np.load(dataset, allow_pickle=True)
    x = payload["x"].astype(np.float32)
    value0 = payload["value0"].astype(np.float32)
    value1 = payload["value1"].astype(np.float32)
    item_class = payload["item_class"].astype(str)
    hook = payload["hook"].astype(str)
    layers, q_weight, q_bias, q_use_bias, expected_obs = load_q_policy(policy)
    if x.shape[1] != expected_obs:
        raise ValueError(f"dataset has {x.shape[1]} features, policy expects {expected_obs}")
    pred0, pred1 = predict_values(x, layers, q_weight, q_bias, q_use_bias, batch_size=8192)
    oracle = (value1 > value0).astype(np.int64)
    predicted = (pred1 > pred0).astype(np.int64)
    best = np.maximum(value0, value1)
    chosen = np.where(predicted == 1, value1, value0)
    regret = best - chosen
    delta = value1 - value0

    groups = defaultdict(list)
    for idx, key in enumerate(zip(item_class, hook)):
        groups[key].append(idx)

    rows = []
    for (cls, hook_name), idxs in groups.items():
        if len(idxs) < min_count:
            continue
        idx = np.asarray(idxs, dtype=np.int64)
        rows.append(
            {
                "item_class": cls,
                "hook": hook_name,
                "decisions": int(len(idx)),
                "oracle_use_rate": float(np.mean(oracle[idx])),
                "model_use_rate": float(np.mean(predicted[idx])),
                "accuracy": float(np.mean(predicted[idx] == oracle[idx])),
                "mean_regret": float(np.mean(regret[idx])),
                "p90_regret": float(np.quantile(regret[idx], 0.90)),
                "mean_delta": float(np.mean(delta[idx])),
            }
        )
    rows.sort(key=lambda row: (row["mean_regret"], 1.0 - row["accuracy"]), reverse=True)

    print(
        f"dataset={dataset} policy={policy} decisions={len(x)} "
        f"accuracy={float(np.mean(predicted == oracle)):.3f} "
        f"mean_regret={float(np.mean(regret)):.4f} "
        f"oracle_use={float(np.mean(oracle)):.3f} model_use={float(np.mean(predicted)):.3f}"
    )
    print("worst item/hook groups:")
    for row in rows[:20]:
        print(
            f"{row['item_class']:<32} {row['hook']:<10} n={row['decisions']:<5} "
            f"acc={row['accuracy']:.3f} regret={row['mean_regret']:.3f} "
            f"use={row['model_use_rate']:.3f}/{row['oracle_use_rate']:.3f}"
        )

    if out:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)
        print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser(description="Analyze an item Q policy on a counterfactual value dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--min-count", type=int, default=100)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    summarize(args.dataset, args.policy, args.min_count, args.out)


if __name__ == "__main__":
    main()
