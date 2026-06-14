import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from policies.replay_features import observation_size
from train_item_model import parse_hidden_sizes


class ReplayQNet(torch.nn.Module):
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
    draw = np.tile(np.asarray([0.0, 1.0], dtype=np.float32), (len(x), 1))
    qx = np.concatenate([np.concatenate([x, skip], axis=1), np.concatenate([x, draw], axis=1)], axis=0)
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
    action = np.concatenate([np.zeros(len(x), dtype=np.int64), np.ones(len(x), dtype=np.int64)], axis=0)
    return qx.astype(np.float32), y, action, {"target_mean": mean, "target_std": std, "source_samples": int(len(x))}


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
        TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), torch.tensor(action)),
        batch_size=batch_size,
        shuffle=False,
    )
    losses = []
    with torch.no_grad():
        for obs, targets, _ in loader:
            obs = obs.to(device)
            targets = targets.to(device)
            losses.append(float(torch.nn.functional.mse_loss(model(obs), targets).detach().cpu()))
    model.train()
    mse = float(np.mean(losses))
    return {"mse": mse, "rmse": float(np.sqrt(mse))}


def train_model(x, y, action, epochs, lr, seed, hidden_sizes, batch_size, device, val_split, weight_decay):
    torch.manual_seed(seed)
    x_train, y_train, action_train, x_val, y_val, action_val = split_train_val(x, y, action, val_split, seed)
    model = ReplayQNet(x.shape[1], hidden_sizes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32), torch.tensor(action_train)),
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
            loss = torch.nn.functional.smooth_l1_loss(model(obs), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val = evaluate_model(model, x_val, y_val, action_val, device, batch_size)
        msg = f"epoch {epoch + 1}/{epochs}: huber={np.mean(losses):.4f}"
        if val is not None:
            msg += f" val_rmse={val['rmse']:.4f}"
            if val["mse"] < best_val:
                best_val = val["mse"]
                best_state = copy.deepcopy(model.state_dict())
        print(msg)
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"restored best validation checkpoint: val_mse={best_val:.4f}")
    return model


def save_policy(model, out, hidden_sizes, metadata):
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
    payload = {
        "type": "replay_q_tanh",
        "observation_size": observation_size(),
        "network_input_size": observation_size() + 2,
        "hidden_sizes": list(hidden_sizes),
        "policy_layers": policy_layers,
        "q_weight": state["q.weight"].detach().cpu().numpy().astype(float).tolist(),
        "q_bias": state["q.bias"].detach().cpu().numpy().astype(float).tolist(),
        "metadata": metadata,
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))


def main():
    parser = argparse.ArgumentParser(description="Train a replay/pass Q(state, action) model.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--hidden-sizes", default="512,512,256")
    parser.add_argument("--weight-decay", type=float, default=0.00001)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--target-scale", default="standard", choices=("none", "standard"))
    parser.add_argument("--seed", type=int, default=12000000)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--out", default="replay_q_policy.json")
    args = parser.parse_args()

    x, y, action, stats = load_q_dataset(args.dataset, args.target_scale)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    print(
        f"training replay Q model: q_samples={len(y)} source_decisions={stats['source_samples']} "
        f"features={x.shape[1]} hidden={hidden_sizes} device={device} source={Path(args.dataset)}"
    )
    model = train_model(x, y, action, args.epochs, args.lr, args.seed, hidden_sizes, args.batch_size, device, args.val_split, args.weight_decay)
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
    }
    save_policy(model, args.out, hidden_sizes, metadata)
    print(f"wrote {args.out}: q_samples={len(y)}")


if __name__ == "__main__":
    main()
