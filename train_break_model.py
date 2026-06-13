import argparse
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from break_env import BreakEnv
from heros import persos_disponibles
from joueurs import Joueur
from monstres import DonjonDeck
from objets import objets_disponibles
from policies import HeuristicPolicy
from policies.break_features import MAX_OBJECTS, extract_break_observation, observation_size
from simu import ordonnanceur


class _ProbeDecision(Exception):
    def __init__(self, state, legal_actions):
        super().__init__("probe break decision")
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

    def choose_item_activation(self, state, legal_actions):
        return self.fallback.choose_item_activation(state, legal_actions)

    def choose_item_to_break(self, state, legal_actions):
        if self.index < len(self.actions):
            action = int(self.actions[self.index])
            self.index += 1
            return action if action in legal_actions else self.fallback.choose_item_to_break(state, legal_actions)
        raise _ProbeDecision(state, legal_actions)


def state_from_replay(env):
    players, objets_simu = env._build_game()
    players[env.controlled_seat].policy = _ProbePolicy(env._actions)
    try:
        ordonnanceur(players, DonjonDeck(), env.pv_min_fuite, objets_simu, False)
    except _ProbeDecision as exc:
        return exc.state, exc.legal_actions
    raise RuntimeError("no pending break decision found")


def collect_examples(samples, seed_start):
    xs = []
    ys = []
    seed = seed_start
    heuristic = HeuristicPolicy("ev")
    while len(xs) < samples:
        env = BreakEnv()
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
            label = int(heuristic.choose_item_to_break(state, legal_actions))
            if 0 <= label < MAX_OBJECTS:
                xs.append(obs.copy())
                ys.append(label)
            obs, _, done, _, _ = env.step(label)
        seed += 1
    return np.vstack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64)


def parse_hidden_sizes(text):
    return [int(part) for part in text.split(",") if part.strip()]


class BreakNet(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super().__init__()
        layers = []
        previous = input_size
        for size in hidden_sizes:
            layers.append(torch.nn.Linear(previous, size))
            layers.append(torch.nn.Tanh())
            previous = size
        self.policy = torch.nn.Sequential(*layers)
        self.action = torch.nn.Linear(previous, MAX_OBJECTS)

    def forward(self, x):
        return self.action(self.policy(x))


def train_model(x, y, epochs, lr, seed, hidden_sizes, batch_size, device):
    torch.manual_seed(seed)
    model = BreakNet(x.shape[1], hidden_sizes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(
        TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=device.type == "cuda",
    )
    counts = np.bincount(y, minlength=MAX_OBJECTS).astype(np.float32)
    weights = counts.sum() / np.maximum(1.0, counts)
    weights = weights / max(1.0, weights.mean())
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    for epoch in range(epochs):
        correct = 0
        losses = []
        seen = 0
        for obs, labels in loader:
            obs = obs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(obs)
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=class_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            correct += int((torch.argmax(logits, dim=1) == labels).sum().detach().cpu())
            seen += int(labels.numel())
        print(f"epoch {epoch + 1}/{epochs}: loss={np.mean(losses):.4f} acc={correct / max(1, seen):.3f}")
    return model


def save_policy(model, out):
    state = model.state_dict()
    payload = {
        "type": "break_actor_tanh",
        "observation_size": observation_size(),
        "max_objects": MAX_OBJECTS,
        "policy_layers": [],
        "action_weight": state["action.weight"].detach().cpu().numpy().astype(float).tolist(),
        "action_bias": state["action.bias"].detach().cpu().numpy().astype(float).tolist(),
    }
    layer_index = 0
    while f"policy.{layer_index}.weight" in state:
        payload["policy_layers"].append({
            "weight": state[f"policy.{layer_index}.weight"].detach().cpu().numpy().astype(float).tolist(),
            "bias": state[f"policy.{layer_index}.bias"].detach().cpu().numpy().astype(float).tolist(),
        })
        layer_index += 2
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))


def main():
    parser = argparse.ArgumentParser(description="Train a supervised object-break model.")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed-start", type=int, default=800000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-sizes", default="256,256,128")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--out", default="break_bc_mlp_policy.json")
    args = parser.parse_args()

    random.seed(args.seed_start)
    x, y = collect_examples(args.samples, args.seed_start)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    print(f"training break model: samples={len(y)} features={x.shape[1]} hidden={hidden_sizes} device={device}")
    model = train_model(x, y, args.epochs, args.lr, args.seed_start, hidden_sizes, args.batch_size, device)
    save_policy(model, args.out)
    print(f"wrote {args.out}: samples={len(y)}")


if __name__ == "__main__":
    main()
