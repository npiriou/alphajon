import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from policies.joint_features import ACTION_FEATURES, FEATURE_VERSION, _action_features, observation_size
from train_item_model import parse_hidden_sizes

MAX_ACTIONS = 4


class JointQNet(torch.nn.Module):
    def __init__(self, state_size, hidden_sizes):
        super().__init__()
        layers = []
        previous = state_size + ACTION_FEATURES
        for size in hidden_sizes:
            layers.append(torch.nn.Linear(previous, size))
            layers.append(torch.nn.Tanh())
            previous = size
        self.policy = torch.nn.Sequential(*layers)
        self.q = torch.nn.Linear(previous, 1)

    def forward_action_features(self, state, action_features):
        x = torch.cat([state, action_features], dim=1)
        return self.q(self.policy(x)).squeeze(1)

    def forward_all(self, state, action_feature_table):
        batch = state.shape[0]
        actions = action_feature_table.shape[1]
        state_rep = state[:, None, :].expand(batch, actions, state.shape[1]).reshape(batch * actions, state.shape[1])
        action_rep = action_feature_table.reshape(batch * actions, action_feature_table.shape[2])
        q = self.forward_action_features(state_rep, action_rep)
        return q.reshape(batch, actions)


def build_action_feature_table(kinds):
    table = np.zeros((len(kinds), MAX_ACTIONS, ACTION_FEATURES), dtype=np.float32)
    for row, kind in enumerate(kinds):
        for action in range(MAX_ACTIONS):
            table[row, action] = _action_features(str(kind), action)
    return table


def load_dataset(path, target_scale):
    payload = np.load(path, allow_pickle=True)
    x = payload["x"].astype(np.float32)
    values = payload["values"].astype(np.float32)
    legal_mask = payload["legal_mask"].astype(bool)
    best = payload["best_action"].astype(np.int64)
    kinds = payload["kind"].astype(object)
    legal_values = values[legal_mask]
    if target_scale == "standard":
        mean = float(np.mean(legal_values))
        std = float(np.std(legal_values))
        if std < 1.0e-6:
            std = 1.0
        y = values.copy()
        y[legal_mask] = (y[legal_mask] - mean) / std
    else:
        mean = 0.0
        std = 1.0
        y = values.copy()
    for row in range(len(y)):
        legal = legal_mask[row]
        fill = float(np.min(y[row, legal])) if bool(np.any(legal)) else 0.0
        y[row, ~legal] = fill
    action_features = build_action_feature_table(kinds)
    return x, y.astype(np.float32), legal_mask, best, action_features, {
        "source_samples": int(len(x)),
        "target_mean": mean,
        "target_std": std,
    }


def split_arrays(arrays, val_split, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(arrays[0]))
    val_size = max(1, int(len(order) * val_split))
    val_idx = order[:val_size]
    train_idx = order[val_size:]
    return [arr[train_idx] for arr in arrays], [arr[val_idx] for arr in arrays]


def evaluate(model, x, y, legal_mask, best, action_features, device, batch_size):
    model.eval()
    loader = DataLoader(
        TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(legal_mask, dtype=torch.bool),
            torch.tensor(best, dtype=torch.long),
            torch.tensor(action_features, dtype=torch.float32),
        ),
        batch_size=batch_size,
    )
    correct = 0
    total = 0
    regrets = []
    losses = []
    with torch.no_grad():
        for state, target, mask, best_action, action_table in loader:
            state = state.to(device)
            target = target.to(device)
            mask = mask.to(device)
            best_action = best_action.to(device)
            action_table = action_table.to(device)
            q = model.forward_all(state, action_table)
            value_loss = torch.nn.functional.smooth_l1_loss(q[mask], target[mask])
            masked_q = q.masked_fill(~mask, -1.0e9)
            pred = torch.argmax(masked_q, dim=1)
            losses.append(float(value_loss.detach().cpu()))
            correct += int((pred == best_action).sum().detach().cpu())
            total += int(best_action.numel())
            chosen = target.gather(1, pred.view(-1, 1)).squeeze(1)
            oracle = target.max(dim=1).values
            regrets.append(float((oracle - chosen).mean().detach().cpu()))
    model.train()
    return {
        "acc": correct / max(1, total),
        "regret": float(np.mean(regrets)),
        "value_loss": float(np.mean(losses)),
    }


def train(x, y, legal_mask, best, action_features, args, device):
    torch.manual_seed(args.seed)
    train_arrays, val_arrays = split_arrays([x, y, legal_mask, best, action_features], args.val_split, args.seed)
    x_train, y_train, mask_train, best_train, af_train = train_arrays
    x_val, y_val, mask_val, best_val, af_val = val_arrays
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    model = JointQNet(x.shape[1], hidden_sizes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(mask_train, dtype=torch.bool),
            torch.tensor(best_train, dtype=torch.long),
            torch.tensor(af_train, dtype=torch.float32),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        pin_memory=device.type == "cuda",
    )
    best_state = None
    best_regret = float("inf")
    for epoch in range(args.epochs):
        losses = []
        for state, target, mask, best_action, action_table in loader:
            state = state.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            best_action = best_action.to(device, non_blocking=True)
            action_table = action_table.to(device, non_blocking=True)
            q = model.forward_all(state, action_table)
            value_loss = torch.nn.functional.smooth_l1_loss(q[mask], target[mask])
            rank_loss = torch.nn.functional.cross_entropy(q.masked_fill(~mask, -1.0e9), best_action)
            loss = value_loss + args.rank_weight * rank_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val = evaluate(model, x_val, y_val, mask_val, best_val, af_val, device, args.batch_size)
        print(
            f"epoch {epoch + 1}/{args.epochs}: loss={np.mean(losses):.4f} "
            f"val_acc={val['acc']:.3f} val_regret={val['regret']:.4f} value={val['value_loss']:.4f}"
        )
        if val["regret"] < best_regret:
            best_regret = val["regret"]
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"restored best validation checkpoint: val_regret={best_regret:.4f}")
    return model, hidden_sizes, best_regret


def save(model, out, hidden_sizes, metadata):
    state = model.state_dict()
    layers = []
    layer_index = 0
    while f"policy.{layer_index}.weight" in state:
        layers.append(
            {
                "weight": state[f"policy.{layer_index}.weight"].detach().cpu().numpy().astype(float).tolist(),
                "bias": state[f"policy.{layer_index}.bias"].detach().cpu().numpy().astype(float).tolist(),
            }
        )
        layer_index += 2
    payload = {
        "type": "joint_decision_q_tanh",
        "feature_version": FEATURE_VERSION,
        "observation_size": observation_size(),
        "state_observation_size": int(state["policy.0.weight"].shape[1] - ACTION_FEATURES),
        "action_features": ACTION_FEATURES,
        "hidden_sizes": list(hidden_sizes),
        "policy_layers": layers,
        "q_weight": state["q.weight"].detach().cpu().numpy().astype(float).tolist(),
        "q_bias": state["q.bias"].detach().cpu().numpy().astype(float).tolist(),
        "metadata": metadata,
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))


def main():
    parser = argparse.ArgumentParser(description="Train a shared Q(state, action) model for item and scry decisions.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-sizes", default="768,512,256")
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--rank-weight", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.00005)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--target-scale", choices=("none", "standard"), default="standard")
    parser.add_argument("--seed", type=int, default=25000000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    x, y, legal_mask, best, action_features, stats = load_dataset(args.dataset, args.target_scale)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    print(
        f"training joint Q: samples={len(x)} state={x.shape[1]} input={x.shape[1] + ACTION_FEATURES} "
        f"device={device}"
    )
    model, hidden_sizes, val_regret = train(x, y, legal_mask, best, action_features, args, device)
    metadata = {
        "source": str(Path(args.dataset)),
        "samples": int(len(x)),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "rank_weight": float(args.rank_weight),
        "weight_decay": float(args.weight_decay),
        "val_split": float(args.val_split),
        "target_scale": args.target_scale,
        "target_mean": stats["target_mean"],
        "target_std": stats["target_std"],
        "val_regret": float(val_regret),
        "device": str(device),
    }
    save(model, args.out, hidden_sizes, metadata)
    print(f"wrote {args.out}: samples={len(x)}")


if __name__ == "__main__":
    main()
