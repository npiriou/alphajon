import argparse
import json
import random

import numpy as np
import torch

from item_env import ItemActivationEnv
from monstres import DonjonDeck
from policies import HeuristicPolicy
from policies.item_features import extract_item_activation_observation, observation_size
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
    def __init__(self, input_size):
        super().__init__()
        self.policy = torch.nn.Sequential(
            torch.nn.Linear(input_size, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
        )
        self.action = torch.nn.Linear(64, 2)

    def forward(self, x):
        return self.action(self.policy(x))


def train_model(x, y, epochs, lr, seed):
    torch.manual_seed(seed)
    model = ItemNet(x.shape[1])
    positives = max(1.0, float(np.sum(y)))
    negatives = max(1.0, float(len(y) - np.sum(y)))
    weights = torch.tensor([1.0, negatives / positives], dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    obs = torch.tensor(x, dtype=torch.float32)
    labels = torch.tensor(y, dtype=torch.long)
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        order = rng.permutation(len(y))
        losses = []
        correct = 0
        for start in range(0, len(y), 256):
            idx = torch.tensor(order[start : start + 256], dtype=torch.long)
            logits = model(obs[idx])
            loss = torch.nn.functional.cross_entropy(logits, labels[idx], weight=weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            correct += int((torch.argmax(logits, dim=1) == labels[idx]).sum().detach())
        print(
            f"epoch {epoch + 1}/{epochs}: loss={np.mean(losses):.4f} "
            f"acc={correct / len(y):.3f} positive={positives / len(y):.3f}"
        )
    return model


def save_policy(model, out):
    state = model.state_dict()
    payload = {
        "type": "item_activation_actor_tanh",
        "observation_size": observation_size(),
        "policy_layers": [
            {
                "weight": state["policy.0.weight"].detach().numpy().astype(float).tolist(),
                "bias": state["policy.0.bias"].detach().numpy().astype(float).tolist(),
            },
            {
                "weight": state["policy.2.weight"].detach().numpy().astype(float).tolist(),
                "bias": state["policy.2.bias"].detach().numpy().astype(float).tolist(),
            },
        ],
        "action_weight": state["action.weight"].detach().numpy().astype(float).tolist(),
        "action_bias": state["action.bias"].detach().numpy().astype(float).tolist(),
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))


def main():
    parser = argparse.ArgumentParser(description="Train a supervised item activation model.")
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed-start", type=int, default=1000000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--out", default="item_bc_mlp_policy.json")
    args = parser.parse_args()

    random.seed(args.seed_start)
    x, y = collect_examples(args.samples, args.seed_start)
    model = train_model(x, y, args.epochs, args.lr, args.seed_start)
    save_policy(model, args.out)
    print(f"wrote {args.out}: samples={len(y)} positive={float(np.mean(y)):.3f}")


if __name__ == "__main__":
    main()
