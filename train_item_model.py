import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from item_env import ItemActivationEnv
from monstres import DonjonDeck
from policies import HeuristicPolicy
from policies.item_features import (
    FEATURE_VERSION,
    LEGACY_FEATURE_VERSION,
    extract_item_activation_observation,
    legacy_observation_size,
    observation_size,
)
from simu import ordonnanceur


class _ProbeDecision(Exception):
    def __init__(self, state, legal_actions):
        super().__init__("probe item activation decision")
        self.state = state
        self.legal_actions = legal_actions


class _ProbePolicy:
    def __init__(self, actions):
        self.actions = list(actions)
        self.index = 0
        self.fallback = HeuristicPolicy("ev")

    def decide_flee(self, state, legal_actions):
        return self.fallback.decide_flee(state, legal_actions)

    def decide_replay(self, state, legal_actions):
        return self.fallback.decide_replay(state, legal_actions)

    def choose_item_to_break(self, state, legal_actions):
        return self.fallback.choose_item_to_break(state, legal_actions)

    def choose_item_activation(self, state, legal_actions):
        if self.index < len(self.actions):
            action = int(self.actions[self.index])
            self.index += 1
            return action if action in legal_actions else 0
        raise _ProbeDecision(state, legal_actions)


def state_from_replay(env):
    players, objets_simu = env._build_game()
    players[env.controlled_seat].policy = _ProbePolicy(env._actions)
    try:
        ordonnanceur(players, DonjonDeck(), env.pv_min_fuite, objets_simu, False)
    except _ProbeDecision as exc:
        return exc.state, exc.legal_actions
    raise RuntimeError("no pending item activation decision found")


def collect_examples(samples, seed_start):
    xs = []
    ys = []
    seed = seed_start
    while len(xs) < samples:
        env = ItemActivationEnv()
        obs, _ = env.reset(seed=seed)
        if env.terminal_players is not None:
            seed += 1
            continue
        done = False
        while not done and len(xs) < samples:
            try:
                state, legal_actions = state_from_replay(env)
            except RuntimeError:
                break
            item = state["item"]
            if state.get("hook") == "en_survie":
                label = 1
            else:
                label = 1 if item.worthit(state["player"], state["card"], state["game"], []) else 0
            xs.append(obs.copy())
            ys.append(label)
            obs, _, done, _, _ = env.step(label)
        seed += 1
    return np.vstack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64)


class ItemNet(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super().__init__()
        layers = []
        previous = input_size
        for size in hidden_sizes:
            layers.append(torch.nn.Linear(previous, size))
            layers.append(torch.nn.Tanh())
            previous = size
        self.policy = torch.nn.Sequential(*layers)
        self.action = torch.nn.Linear(previous, 2)

    def forward(self, x):
        return self.action(self.policy(x))


def parse_hidden_sizes(text):
    return [int(part) for part in text.split(",") if part.strip()]


def load_dataset(path):
    payload = np.load(path, allow_pickle=True)
    return payload["x"].astype(np.float32), payload["y"].astype(np.int64)


def split_train_val(x, y, val_split, seed):
    if val_split <= 0:
        return x, y, None, None
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    val_size = max(1, int(len(y) * val_split))
    val_idx = order[:val_size]
    train_idx = order[val_size:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def evaluate_model(model, x, y, device, batch_size):
    if x is None or y is None or len(y) == 0:
        return None
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=False,
    )
    total = 0
    correct = 0
    losses = []
    with torch.no_grad():
        for obs, labels in loader:
            obs = obs.to(device)
            labels = labels.to(device)
            logits = model(obs)
            losses.append(float(torch.nn.functional.cross_entropy(logits, labels).detach().cpu()))
            correct += int((torch.argmax(logits, dim=1) == labels).sum().detach().cpu())
            total += int(labels.numel())
    model.train()
    return {"loss": float(np.mean(losses)), "acc": correct / max(1, total)}


def train_model(x, y, epochs, lr, seed, hidden_sizes, batch_size, device, val_split, weight_decay, class_weight):
    torch.manual_seed(seed)
    x_train, y_train, x_val, y_val = split_train_val(x, y, val_split, seed)
    model = ItemNet(x.shape[1], hidden_sizes).to(device)
    positives = max(1.0, float(np.sum(y)))
    negatives = max(1.0, float(len(y) - np.sum(y)))
    weights = None
    if class_weight == "balanced":
        weights = torch.tensor([1.0, negatives / positives], dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=device.type == "cuda",
    )
    best_state = None
    best_val_acc = -1.0
    best_val_loss = float("inf")
    for epoch in range(epochs):
        losses = []
        correct = 0
        seen = 0
        for obs, labels in loader:
            obs = obs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(obs)
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            correct += int((torch.argmax(logits, dim=1) == labels).sum().detach().cpu())
            seen += int(labels.numel())
        val = evaluate_model(model, x_val, y_val, device, batch_size)
        msg = (
            f"epoch {epoch + 1}/{epochs}: loss={np.mean(losses):.4f} "
            f"acc={correct / max(1, seen):.3f} positive={positives / len(y):.3f}"
        )
        if val is not None:
            msg += f" val_loss={val['loss']:.4f} val_acc={val['acc']:.3f}"
            if val["acc"] > best_val_acc or (val["acc"] == best_val_acc and val["loss"] < best_val_loss):
                best_val_acc = val["acc"]
                best_val_loss = val["loss"]
                best_state = copy.deepcopy(model.state_dict())
        print(msg)
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"restored best validation checkpoint: val_loss={best_val_loss:.4f} val_acc={best_val_acc:.3f}")
    return model


def save_policy(model, out, hidden_sizes, metadata=None):
    state = model.state_dict()
    input_size = int(state["policy.0.weight"].shape[1])
    if input_size == observation_size():
        feature_version = FEATURE_VERSION
    elif input_size == legacy_observation_size():
        feature_version = LEGACY_FEATURE_VERSION
    else:
        feature_version = "unknown"
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
        "type": "item_activation_actor_tanh",
        "feature_version": feature_version,
        "observation_size": input_size,
        "hidden_sizes": list(hidden_sizes),
        "policy_layers": policy_layers,
        "action_weight": state["action.weight"].detach().cpu().numpy().astype(float).tolist(),
        "action_bias": state["action.bias"].detach().cpu().numpy().astype(float).tolist(),
    }
    if metadata:
        payload["metadata"] = metadata
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))


def main():
    parser = argparse.ArgumentParser(description="Train a supervised item activation model.")
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed-start", type=int, default=1000000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-sizes", default="256,256,128")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--class-weight", default="none", choices=("none", "balanced"))
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--out", default="item_bc_mlp_policy.json")
    args = parser.parse_args()

    random.seed(args.seed_start)
    if args.dataset:
        x, y = load_dataset(args.dataset)
        source = str(Path(args.dataset))
    else:
        x, y = collect_examples(args.samples, args.seed_start)
        source = "online"
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    print(
        f"training item model: samples={len(y)} features={x.shape[1]} "
        f"hidden={hidden_sizes} device={device} source={source}"
    )
    model = train_model(
        x,
        y,
        args.epochs,
        args.lr,
        args.seed_start,
        hidden_sizes,
        args.batch_size,
        device,
        args.val_split,
        args.weight_decay,
        args.class_weight,
    )
    metadata = {
        "source": source,
        "samples": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "seed_start": int(args.seed_start),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "class_weight": args.class_weight,
        "val_split": float(args.val_split),
        "device": str(device),
    }
    save_policy(model, args.out, hidden_sizes, metadata=metadata)
    print(f"wrote {args.out}: samples={len(y)} positive={float(np.mean(y)):.3f}")


if __name__ == "__main__":
    main()
