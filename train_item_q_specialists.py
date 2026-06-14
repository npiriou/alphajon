import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from analyze_item_q_policy import load_q_policy, predict_values
from train_item_model import parse_hidden_sizes


class SpecialistQNet(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super().__init__()
        layers = []
        previous = input_size + 2
        for size in hidden_sizes:
            layers.append(torch.nn.Linear(previous, size))
            layers.append(torch.nn.Tanh())
            previous = size
        self.policy = torch.nn.Sequential(*layers)
        self.q = torch.nn.Linear(previous, 1)

    def forward_action(self, obs, action_id):
        action = torch.zeros((obs.shape[0], 2), dtype=obs.dtype, device=obs.device)
        action[:, action_id] = 1.0
        x = torch.cat([obs, action], dim=1)
        return self.q(self.policy(x)).squeeze(-1)

    def forward_pair(self, obs):
        return self.forward_action(obs, 0), self.forward_action(obs, 1)


def group_indices(item_class, hook):
    groups = defaultdict(list)
    for idx, (cls, hook_name) in enumerate(zip(item_class.astype(str), hook.astype(str))):
        groups[f"{cls}|{hook_name}"].append(idx)
    return {key: np.asarray(value, dtype=np.int64) for key, value in groups.items()}


def regret_for_predictions(pred0, pred1, value0, value1):
    predicted = pred1 > pred0
    best = np.maximum(value0, value1)
    chosen = np.where(predicted, value1, value0)
    return best - chosen, predicted


def select_groups(dataset, base_policy, min_count, top_k):
    payload = np.load(dataset, allow_pickle=True)
    x = payload["x"].astype(np.float32)
    value0 = payload["value0"].astype(np.float32)
    value1 = payload["value1"].astype(np.float32)
    item_class = payload["item_class"]
    hook = payload["hook"]
    layers, q_weight, q_bias, q_use_bias, expected_obs = load_q_policy(base_policy)
    if x.shape[1] != expected_obs:
        raise ValueError(f"dataset has {x.shape[1]} features, policy expects {expected_obs}")
    pred0, pred1 = predict_values(x, layers, q_weight, q_bias, q_use_bias, batch_size=8192)
    regret, predicted = regret_for_predictions(pred0, pred1, value0, value1)
    oracle = value1 > value0
    groups = group_indices(item_class, hook)
    rows = []
    for key, idx in groups.items():
        if len(idx) < min_count:
            continue
        rows.append(
            {
                "key": key,
                "idx": idx,
                "count": int(len(idx)),
                "mean_regret": float(np.mean(regret[idx])),
                "accuracy": float(np.mean(predicted[idx] == oracle[idx])),
                "model_use": float(np.mean(predicted[idx])),
                "oracle_use": float(np.mean(oracle[idx])),
            }
        )
    rows.sort(key=lambda row: (row["mean_regret"], 1.0 - row["accuracy"]), reverse=True)
    return payload, rows[:top_k]


def prepare_targets(value0, value1, target_scale):
    y = np.stack([value0, value1], axis=1).astype(np.float32)
    if target_scale == "standard":
        mean = float(np.mean(y))
        std = float(np.std(y))
        if std < 1.0e-6:
            std = 1.0
        y = ((y - mean) / std).astype(np.float32)
    else:
        mean = 0.0
        std = 1.0
    delta = y[:, 1] - y[:, 0]
    label = (delta > 0).astype(np.float32)
    weight = np.clip(np.abs(delta), 0.05, 4.0).astype(np.float32)
    return y, label, weight, mean, std


def split(x, y, label, weight, val_split, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(label))
    val_size = max(1, int(len(label) * val_split))
    val_idx = order[:val_size]
    train_idx = order[val_size:]
    return x[train_idx], y[train_idx], label[train_idx], weight[train_idx], x[val_idx], y[val_idx], label[val_idx], weight[val_idx]


def evaluate(model, x, y, label, device, batch_size):
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    regrets = []
    correct = 0
    total = 0
    with torch.no_grad():
        for obs, targets, labels in loader:
            obs = obs.to(device)
            targets = targets.to(device)
            labels = labels.to(device)
            q0, q1 = model.forward_pair(obs)
            pred = q1 > q0
            chosen = torch.where(pred, targets[:, 1], targets[:, 0])
            best = torch.maximum(targets[:, 0], targets[:, 1])
            regrets.append(float((best - chosen).mean().detach().cpu()))
            correct += int((pred == (labels > 0.5)).sum().detach().cpu())
            total += int(labels.numel())
    model.train()
    return {"regret": float(np.mean(regrets)), "acc": correct / max(1, total)}


def train_one(x, y, label, weight, hidden_sizes, epochs, batch_size, lr, weight_decay, rank_weight, val_split, seed, device):
    x_train, y_train, label_train, weight_train, x_val, y_val, label_val, _ = split(x, y, label, weight, val_split, seed)
    model = SpecialistQNet(x.shape[1], hidden_sizes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(label_train, dtype=torch.float32),
            torch.tensor(weight_train, dtype=torch.float32),
        ),
        batch_size=min(batch_size, len(label_train)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=device.type == "cuda",
    )
    best_state = None
    best_regret = float("inf")
    for _ in range(epochs):
        for obs, targets, labels, weights in loader:
            obs = obs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            q0, q1 = model.forward_pair(obs)
            value_loss = torch.nn.functional.smooth_l1_loss(torch.stack([q0, q1], dim=1), targets)
            rank_loss = torch.nn.functional.binary_cross_entropy_with_logits(q1 - q0, labels, reduction="none")
            loss = value_loss + rank_weight * (rank_loss * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val = evaluate(model, x_val, y_val, label_val, device, batch_size)
        if val["regret"] < best_regret:
            best_regret = val["regret"]
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, evaluate(model, x_val, y_val, label_val, device, batch_size)


def export_specialist(model):
    state = model.state_dict()
    policy_layers = []
    layer_index = 0
    while f"policy.{layer_index}.weight" in state:
        policy_layers.append(
            {
                "weight": state[f"policy.{layer_index}.weight"].detach().cpu().numpy().astype(float).tolist(),
                "bias": state[f"policy.{layer_index}.bias"].detach().cpu().numpy().astype(float).tolist(),
            }
        )
        layer_index += 2
    return {
        "policy_layers": policy_layers,
        "q_weight": state["q.weight"].detach().cpu().numpy().astype(float).tolist(),
        "q_bias": state["q.bias"].detach().cpu().numpy().astype(float).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Train item/hook specialist Q heads for worst-regret items.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base-policy", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--min-count", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-sizes", default="128,64")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.00001)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--target-scale", default="standard", choices=("none", "standard"))
    parser.add_argument("--seed", type=int, default=15000000)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    payload, groups = select_groups(args.dataset, args.base_policy, args.min_count, args.top_k)
    x_all = payload["x"].astype(np.float32)
    value0_all = payload["value0"].astype(np.float32)
    value1_all = payload["value1"].astype(np.float32)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)

    with open(args.base_policy, "r", encoding="utf-8") as fh:
        policy_payload = json.load(fh)
    specialists = policy_payload.setdefault("metadata", {}).setdefault("item_hook_specialists", {})
    report = []
    for rank, row in enumerate(groups, start=1):
        idx = row["idx"]
        x = x_all[idx]
        y, label, weight, mean, std = prepare_targets(value0_all[idx], value1_all[idx], args.target_scale)
        model, val = train_one(
            x,
            y,
            label,
            weight,
            hidden_sizes,
            args.epochs,
            args.batch_size,
            args.lr,
            args.weight_decay,
            args.rank_weight,
            args.val_split,
            args.seed + rank,
            device,
        )
        specialists[row["key"]] = export_specialist(model)
        specialists[row["key"]]["metadata"] = {
            "source": str(Path(args.dataset)),
            "count": row["count"],
            "base_mean_regret": row["mean_regret"],
            "base_accuracy": row["accuracy"],
            "val_regret": val["regret"],
            "val_accuracy": val["acc"],
            "target_scale": args.target_scale,
            "target_mean": mean,
            "target_std": std,
            "hidden_sizes": hidden_sizes,
        }
        report.append((row["key"], row["count"], row["mean_regret"], row["accuracy"], val["regret"], val["acc"]))
        print(
            f"{rank:02d}/{len(groups)} {row['key']}: n={row['count']} "
            f"base_regret={row['mean_regret']:.3f} base_acc={row['accuracy']:.3f} "
            f"val_regret={val['regret']:.3f} val_acc={val['acc']:.3f}"
        )

    policy_payload["metadata"]["item_hook_specialist_training"] = {
        "dataset": str(Path(args.dataset)),
        "base_policy": str(Path(args.base_policy)),
        "top_k": int(args.top_k),
        "min_count": int(args.min_count),
        "epochs": int(args.epochs),
        "hidden_sizes": hidden_sizes,
        "rank_weight": float(args.rank_weight),
        "specialists": [row[0] for row in report],
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(policy_payload, fh, separators=(",", ":"))
    print(f"wrote {args.out}: specialists={len(report)}")


if __name__ == "__main__":
    main()
