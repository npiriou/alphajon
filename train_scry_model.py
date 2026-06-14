import argparse
import copy
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from policies.scry_features import observation_size
from train_item_model import parse_hidden_sizes


class ScryNet(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, actions=4):
        super().__init__()
        layers = []
        previous = input_size
        for size in hidden_sizes:
            layers.append(torch.nn.Linear(previous, size))
            layers.append(torch.nn.Tanh())
            previous = size
        self.policy = torch.nn.Sequential(*layers)
        self.q = torch.nn.Linear(previous, actions)

    def forward(self, x):
        return self.q(self.policy(x))


def split(x, values, best_action, val_split, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(best_action))
    val_size = max(1, int(len(best_action) * val_split))
    val_idx = order[:val_size]
    train_idx = order[val_size:]
    return x[train_idx], values[train_idx], best_action[train_idx], x[val_idx], values[val_idx], best_action[val_idx]


def evaluate(model, x, values, best_action, device, batch_size):
    model.eval()
    loader = DataLoader(TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(values, dtype=torch.float32), torch.tensor(best_action, dtype=torch.long)), batch_size=batch_size)
    correct = 0
    total = 0
    regrets = []
    with torch.no_grad():
        for obs, target_values, best in loader:
            obs = obs.to(device)
            target_values = target_values.to(device)
            best = best.to(device)
            q = model(obs)
            pred = torch.argmax(q, dim=1)
            correct += int((pred == best).sum().detach().cpu())
            total += int(best.numel())
            chosen = target_values.gather(1, pred.view(-1, 1)).squeeze(1)
            oracle = torch.max(target_values, dim=1).values
            regrets.append(float((oracle - chosen).mean().detach().cpu()))
    model.train()
    return {"acc": correct / max(1, total), "regret": float(np.mean(regrets))}


def save(model, out, hidden_sizes, metadata):
    state = model.state_dict()
    policy_layers = []
    layer_index = 0
    while f"policy.{layer_index}.weight" in state:
        policy_layers.append({
            "weight": state[f"policy.{layer_index}.weight"].detach().cpu().numpy().astype(float).tolist(),
            "bias": state[f"policy.{layer_index}.bias"].detach().cpu().numpy().astype(float).tolist(),
        })
        layer_index += 2
    payload = {
        "type": "scry_window_q_tanh",
        "observation_size": observation_size(),
        "hidden_sizes": list(hidden_sizes),
        "policy_layers": policy_layers,
        "q_weight": state["q.weight"].detach().cpu().numpy().astype(float).tolist(),
        "q_bias": state["q.bias"].detach().cpu().numpy().astype(float).tolist(),
        "metadata": metadata,
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-sizes", default="128,128")
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.00001)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=21000000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    payload = np.load(args.dataset, allow_pickle=True)
    x = payload["x"].astype(np.float32)
    values = payload["values"].astype(np.float32)
    best_action = payload["best_action"].astype(np.int64)
    legal_mask = values > -1.0e8
    target_mean = float(np.mean(values[legal_mask]))
    target_std = float(np.std(values[legal_mask]))
    if target_std < 1.0e-6:
        target_std = 1.0
    train_values = values.copy()
    train_values[legal_mask] = (train_values[legal_mask] - target_mean) / target_std
    for row_idx in range(train_values.shape[0]):
        row_legal = legal_mask[row_idx]
        fill = float(np.min(train_values[row_idx, row_legal])) if bool(np.any(row_legal)) else 0.0
        train_values[row_idx, ~row_legal] = fill
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    x_train, v_train, a_train, x_val, v_val, a_val = split(x, train_values, best_action, args.val_split, args.seed)
    model = ScryNet(x.shape[1], hidden_sizes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(v_train, dtype=torch.float32),
            torch.tensor(a_train, dtype=torch.long),
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
        for obs, target_values, labels in loader:
            obs = obs.to(device, non_blocking=True)
            target_values = target_values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(obs)
            value_loss = torch.nn.functional.smooth_l1_loss(logits, target_values)
            rank_loss = torch.nn.functional.cross_entropy(logits, labels)
            loss = args.value_weight * value_loss + args.rank_weight * rank_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val = evaluate(model, x_val, v_val, a_val, device, args.batch_size)
        print(f"epoch {epoch + 1}/{args.epochs}: loss={np.mean(losses):.4f} val_acc={val['acc']:.3f} val_regret={val['regret']:.4f}")
        if val["regret"] < best_regret:
            best_regret = val["regret"]
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    metadata = {
        "source": args.dataset,
        "samples": int(len(x)),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "hidden_sizes": hidden_sizes,
        "device": str(device),
        "val_regret": best_regret,
        "target_mean": target_mean,
        "target_std": target_std,
    }
    save(model, args.out, hidden_sizes, metadata)
    print(f"wrote {args.out}: samples={len(x)}")


if __name__ == "__main__":
    main()
