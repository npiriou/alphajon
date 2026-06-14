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
        previous = input_size
        for size in hidden_sizes:
            layers.append(torch.nn.Linear(previous, size))
            layers.append(torch.nn.Tanh())
            previous = size
        self.policy = torch.nn.Sequential(*layers)
        self.q = torch.nn.Linear(previous, 1)

    def forward(self, x):
        return self.q(self.policy(x)).squeeze(-1)


def load_q_dataset(path, target_scale):
    payload = np.load(path, allow_pickle=True)
    x = payload["x"].astype(np.float32)
    v0 = payload["value0"].astype(np.float32)
    v1 = payload["value1"].astype(np.float32)
    skip = np.tile(np.asarray([1.0, 0.0], dtype=np.float32), (len(x), 1))
    use = np.tile(np.asarray([0.0, 1.0], dtype=np.float32), (len(x), 1))
    qx = np.concatenate(
        [
            np.concatenate([x, skip], axis=1),
            np.concatenate([x, use], axis=1),
        ],
        axis=0,
    ).astype(np.float32)
    y = np.concatenate([v0, v1], axis=0).astype(np.float32)
    if target_scale == "standard":
        mean = float(np.mean(y))
        std = float(np.std(y))
        if std < 1.0e-6:
            std = 1.0
        y = ((y - mean) / std).astype(np.float32)
    else:
        mean = 0.0
        std = 1.0
    action = np.concatenate(
        [np.zeros(len(x), dtype=np.int64), np.ones(len(x), dtype=np.int64)],
        axis=0,
    )
    return qx, y, action, {"target_mean": mean, "target_std": std, "source_samples": int(len(x))}


def split_train_val(x, y, action, val_split, seed):
    if val_split <= 0:
        return x, y, action, None, None, None
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    val_size = max(1, int(len(y) * val_split))
    val_idx = order[:val_size]
    train_idx = order[val_size:]
    return x[train_idx], y[train_idx], action[train_idx], x[val_idx], y[val_idx], action[val_idx]


def evaluate_model(model, x, y, action, device, batch_size):
    if x is None or y is None or len(y) == 0:
        return None
    model.eval()
    loader = DataLoader(
        TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(action, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    losses = []
    by_action = {0: [], 1: []}
    with torch.no_grad():
        for obs, targets, actions in loader:
            obs = obs.to(device)
            targets = targets.to(device)
            pred = model(obs)
            error = torch.nn.functional.mse_loss(pred, targets, reduction="none")
            losses.append(float(error.mean().detach().cpu()))
            for action_id in (0, 1):
                mask = actions == action_id
                if bool(mask.any()):
                    by_action[action_id].append(float(error[mask.to(device)].mean().detach().cpu()))
    model.train()
    return {
        "mse": float(np.mean(losses)),
        "rmse": float(np.sqrt(np.mean(losses))),
        "mse_skip": float(np.mean(by_action[0])) if by_action[0] else 0.0,
        "mse_use": float(np.mean(by_action[1])) if by_action[1] else 0.0,
    }


def train_model(x, y, action, epochs, lr, seed, hidden_sizes, batch_size, device, val_split, weight_decay):
    torch.manual_seed(seed)
    x_train, y_train, action_train, x_val, y_val, action_val = split_train_val(x, y, action, val_split, seed)
    model = ItemQNet(x.shape[1], hidden_sizes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(action_train, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=device.type == "cuda",
    )
    best_state = None
    best_val = float("inf")
    for epoch in range(epochs):
        losses = []
        for obs, targets, _ in loader:
            obs = obs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            pred = model(obs)
            loss = torch.nn.functional.smooth_l1_loss(pred, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val = evaluate_model(model, x_val, y_val, action_val, device, batch_size)
        msg = f"epoch {epoch + 1}/{epochs}: huber={np.mean(losses):.4f}"
        if val is not None:
            msg += (
                f" val_rmse={val['rmse']:.4f} "
                f"skip_mse={val['mse_skip']:.4f} use_mse={val['mse_use']:.4f}"
            )
            if val["mse"] < best_val:
                best_val = val["mse"]
                best_state = copy.deepcopy(model.state_dict())
        print(msg)
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"restored best validation checkpoint: val_mse={best_val:.4f}")
    return model


def save_policy(model, out, hidden_sizes, metadata, base_observation_size):
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
    if base_observation_size == observation_size():
        feature_version = FEATURE_VERSION
    elif base_observation_size == v2_observation_size():
        feature_version = V2_FEATURE_VERSION
    else:
        feature_version = "unknown"
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
    parser = argparse.ArgumentParser(description="Train an item activation Q(state, action) model.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--hidden-sizes", default="768,512,256")
    parser.add_argument("--weight-decay", type=float, default=0.00001)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--target-scale", default="standard", choices=("none", "standard"))
    parser.add_argument("--seed", type=int, default=10000000)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--out", default="item_q_policy.json")
    args = parser.parse_args()

    x, y, action, stats = load_q_dataset(args.dataset, args.target_scale)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    print(
        f"training item Q model: q_samples={len(y)} source_decisions={stats['source_samples']} "
        f"features={x.shape[1]} hidden={hidden_sizes} device={device} source={Path(args.dataset)}"
    )
    model = train_model(
        x,
        y,
        action,
        args.epochs,
        args.lr,
        args.seed,
        hidden_sizes,
        args.batch_size,
        device,
        args.val_split,
        args.weight_decay,
    )
    metadata = {
        "source": str(Path(args.dataset)),
        "source_decisions": stats["source_samples"],
        "q_samples": int(len(y)),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "val_split": float(args.val_split),
        "target_scale": args.target_scale,
        "target_mean": stats["target_mean"],
        "target_std": stats["target_std"],
        "device": str(device),
        "force_survival": True,
    }
    save_policy(model, args.out, hidden_sizes, metadata, x.shape[1] - 2)
    print(f"wrote {args.out}: q_samples={len(y)}")


if __name__ == "__main__":
    main()
