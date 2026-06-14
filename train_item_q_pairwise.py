import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from policies.item_features import FEATURE_VERSION, V2_FEATURE_VERSION, observation_size, v2_observation_size
from train_item_model import parse_hidden_sizes


class ItemQNet(torch.nn.Module):
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


def load_dataset(path, target_scale, min_abs_delta):
    payload = np.load(path, allow_pickle=True)
    x = payload["x"].astype(np.float32)
    v0 = payload["value0"].astype(np.float32)
    v1 = payload["value1"].astype(np.float32)
    delta = v1 - v0
    if min_abs_delta > 0:
        keep = np.abs(delta) >= min_abs_delta
        x = x[keep]
        v0 = v0[keep]
        v1 = v1[keep]
        delta = delta[keep]
    y = np.stack([v0, v1], axis=1).astype(np.float32)
    if target_scale == "standard":
        mean = float(np.mean(y))
        std = float(np.std(y))
        if std < 1.0e-6:
            std = 1.0
        y = ((y - mean) / std).astype(np.float32)
        delta = (y[:, 1] - y[:, 0]).astype(np.float32)
    else:
        mean = 0.0
        std = 1.0
    label = (delta > 0).astype(np.float32)
    weight = np.clip(np.abs(delta), 0.05, 4.0).astype(np.float32)
    return x, y, label, weight, {
        "target_mean": mean,
        "target_std": std,
        "source_samples": int(len(payload["x"])),
        "kept_samples": int(len(x)),
        "positive_rate": float(np.mean(label)) if len(label) else 0.0,
    }


def split_train_val(x, y, label, weight, val_split, seed):
    if val_split <= 0:
        return x, y, label, weight, None, None, None, None
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(label))
    val_size = max(1, int(len(label) * val_split))
    val_idx = order[:val_size]
    train_idx = order[val_size:]
    return (
        x[train_idx],
        y[train_idx],
        label[train_idx],
        weight[train_idx],
        x[val_idx],
        y[val_idx],
        label[val_idx],
        weight[val_idx],
    )


def evaluate_model(model, x, y, label, weight, device, batch_size):
    if x is None or y is None or len(label) == 0:
        return None
    model.eval()
    loader = DataLoader(
        TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(weight, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    value_losses = []
    rank_losses = []
    correct = 0
    total = 0
    regrets = []
    with torch.no_grad():
        for obs, targets, labels, weights in loader:
            obs = obs.to(device)
            targets = targets.to(device)
            labels = labels.to(device)
            weights = weights.to(device)
            q0, q1 = model.forward_pair(obs)
            value_losses.append(float(torch.nn.functional.smooth_l1_loss(torch.stack([q0, q1], dim=1), targets).detach().cpu()))
            logits = q1 - q0
            rank_losses.append(
                float((torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none") * weights).mean().detach().cpu())
            )
            pred = q1 > q0
            correct += int((pred == (labels > 0.5)).sum().detach().cpu())
            total += int(labels.numel())
            chosen = torch.where(pred, targets[:, 1], targets[:, 0])
            best = torch.maximum(targets[:, 0], targets[:, 1])
            regrets.append(float((best - chosen).mean().detach().cpu()))
    model.train()
    return {
        "value_loss": float(np.mean(value_losses)),
        "rank_loss": float(np.mean(rank_losses)),
        "acc": correct / max(1, total),
        "regret": float(np.mean(regrets)),
    }


def predict_pair(model, x, device, batch_size):
    model.eval()
    loader = DataLoader(TensorDataset(torch.tensor(x, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    pred0 = []
    pred1 = []
    with torch.no_grad():
        for (obs,) in loader:
            obs = obs.to(device)
            q0, q1 = model.forward_pair(obs)
            pred0.append(q0.detach().cpu().numpy())
            pred1.append(q1.detach().cpu().numpy())
    model.train()
    return np.concatenate(pred0), np.concatenate(pred1)


def calibrate_use_bias(model, x, y, device, batch_size):
    pred0, pred1 = predict_pair(model, x, device, batch_size)
    margins = pred0 - pred1
    candidates = np.unique(np.quantile(margins, np.linspace(0.0, 1.0, 401))).astype(np.float32)
    candidates = np.concatenate([candidates, np.asarray([0.0], dtype=np.float32)])
    best_bias = 0.0
    best_regret = float("inf")
    best_use = 0.0
    best_acc = 0.0
    oracle = y[:, 1] > y[:, 0]
    best = np.maximum(y[:, 0], y[:, 1])
    for bias in candidates:
        use = (pred1 + bias) > pred0
        chosen = np.where(use, y[:, 1], y[:, 0])
        regret = float(np.mean(best - chosen))
        if regret < best_regret:
            best_regret = regret
            best_bias = float(bias)
            best_use = float(np.mean(use))
            best_acc = float(np.mean(use == oracle))
    return {
        "q_use_bias": best_bias,
        "calibrated_regret": best_regret,
        "calibrated_use_rate": best_use,
        "calibrated_accuracy": best_acc,
    }


def train_model(x, y, label, weight, epochs, lr, seed, hidden_sizes, batch_size, device, val_split, weight_decay, rank_weight):
    torch.manual_seed(seed)
    split = split_train_val(x, y, label, weight, val_split, seed)
    x_train, y_train, label_train, weight_train, x_val, y_val, label_val, weight_val = split
    model = ItemQNet(x.shape[1], hidden_sizes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(label_train, dtype=torch.float32),
            torch.tensor(weight_train, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=device.type == "cuda",
    )
    best_state = None
    best_score = float("inf")
    for epoch in range(epochs):
        losses = []
        for obs, targets, labels, weights in loader:
            obs = obs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            q0, q1 = model.forward_pair(obs)
            pred = torch.stack([q0, q1], dim=1)
            value_loss = torch.nn.functional.smooth_l1_loss(pred, targets)
            rank_loss = torch.nn.functional.binary_cross_entropy_with_logits(q1 - q0, labels, reduction="none")
            loss = value_loss + rank_weight * (rank_loss * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val = evaluate_model(model, x_val, y_val, label_val, weight_val, device, batch_size)
        msg = f"epoch {epoch + 1}/{epochs}: loss={np.mean(losses):.4f}"
        if val is not None:
            score = val["regret"]
            msg += (
                f" val_acc={val['acc']:.3f} val_regret={val['regret']:.4f} "
                f"value={val['value_loss']:.4f} rank={val['rank_loss']:.4f}"
            )
            if score < best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
        print(msg)
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"restored best validation checkpoint: val_regret={best_score:.4f}")
    return model


def save_policy(model, out, hidden_sizes, metadata, base_observation_size):
    state = model.state_dict()
    feature_version = FEATURE_VERSION if base_observation_size == observation_size() else V2_FEATURE_VERSION
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
    payload = {
        "type": "item_activation_q_tanh",
        "feature_version": feature_version,
        "observation_size": int(base_observation_size),
        "network_input_size": int(base_observation_size) + 2,
        "hidden_sizes": list(hidden_sizes),
        "policy_layers": policy_layers,
        "q_weight": state["q.weight"].detach().cpu().numpy().astype(float).tolist(),
        "q_bias": state["q.bias"].detach().cpu().numpy().astype(float).tolist(),
        "metadata": metadata,
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))


def main():
    parser = argparse.ArgumentParser(description="Train item Q with value regression plus pairwise ranking.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0004)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--hidden-sizes", default="768,512,256")
    parser.add_argument("--weight-decay", type=float, default=0.00001)
    parser.add_argument("--rank-weight", type=float, default=0.75)
    parser.add_argument("--min-abs-delta", type=float, default=0.0)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--target-scale", default="standard", choices=("none", "standard"))
    parser.add_argument("--calibrate-use-bias", action="store_true")
    parser.add_argument("--seed", type=int, default=14000000)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--out", default="item_q_pairwise_policy.json")
    args = parser.parse_args()

    x, y, label, weight, stats = load_dataset(args.dataset, args.target_scale, args.min_abs_delta)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    print(
        f"training item pairwise Q: samples={len(label)} source={stats['source_samples']} "
        f"positive={stats['positive_rate']:.3f} features={x.shape[1] + 2} hidden={hidden_sizes} device={device}"
    )
    model = train_model(
        x,
        y,
        label,
        weight,
        args.epochs,
        args.lr,
        args.seed,
        hidden_sizes,
        args.batch_size,
        device,
        args.val_split,
        args.weight_decay,
        args.rank_weight,
    )
    metadata = {
        "source": str(Path(args.dataset)),
        "source_decisions": stats["source_samples"],
        "kept_decisions": stats["kept_samples"],
        "positive_rate": stats["positive_rate"],
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "rank_weight": float(args.rank_weight),
        "min_abs_delta": float(args.min_abs_delta),
        "val_split": float(args.val_split),
        "target_scale": args.target_scale,
        "target_mean": stats["target_mean"],
        "target_std": stats["target_std"],
        "device": str(device),
        "force_survival": True,
    }
    if args.calibrate_use_bias:
        calibration = calibrate_use_bias(model, x, y, device, args.batch_size)
        metadata.update(calibration)
        print(
            "calibrated q_use_bias="
            f"{calibration['q_use_bias']:.4f} regret={calibration['calibrated_regret']:.4f} "
            f"acc={calibration['calibrated_accuracy']:.3f} use={calibration['calibrated_use_rate']:.3f}"
        )
    save_policy(model, args.out, hidden_sizes, metadata, x.shape[1])
    print(f"wrote {args.out}: decisions={len(label)}")


if __name__ == "__main__":
    main()
