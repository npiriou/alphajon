import argparse
import json
import random

import numpy as np

from flee_env import FleeEnv
from policies import FLEE_ACTION_ATTEMPT, HeuristicPolicy
from policies.flee_features import observation_size, sigmoid


def collect_examples(samples, seed_start):
    xs = []
    ys = []
    heuristic = HeuristicPolicy("ev")
    seed = seed_start
    while len(xs) < samples:
        env = FleeEnv()
        obs, info = env.reset(seed=seed)
        if env.terminal_players is not None:
            seed += 1
            continue
        done = False
        while not done and len(xs) < samples:
            try:
                state = _state_from_replay(env)
            except RuntimeError:
                break
            label = heuristic.decide_flee(state, (0, 1))
            xs.append(obs.copy())
            ys.append(1.0 if label == FLEE_ACTION_ATTEMPT else 0.0)
            obs, _, done, _, info = env.step(label)
        seed += 1
    return np.vstack(xs), np.asarray(ys, dtype=np.float32)


def _state_from_replay(env):
    players, objets_simu = env._build_game()
    probe = _ProbePolicy(env._actions)
    players[env.controlled_seat].policy = probe
    try:
        from monstres import DonjonDeck
        from simu import ordonnanceur

        ordonnanceur(players, DonjonDeck(), env.pv_min_fuite, objets_simu, False)
    except _ProbeDecision as exc:
        return exc.state
    raise RuntimeError("no pending decision found while collecting examples")


class _ProbeDecision(Exception):
    def __init__(self, state):
        super().__init__("probe decision")
        self.state = state


class _ProbePolicy:
    def __init__(self, actions):
        self.actions = list(actions)
        self.index = 0

    def decide_flee(self, state, legal_actions):
        if self.index < len(self.actions):
            action = int(self.actions[self.index])
            self.index += 1
            return action if action in legal_actions else 0
        raise _ProbeDecision(state)


def train_logistic(x, y, epochs, lr, l2):
    rng = np.random.default_rng(12345)
    weights = rng.normal(0.0, 0.01, size=x.shape[1]).astype(np.float32)
    bias = 0.0
    n = x.shape[0]
    positives = max(1.0, float(np.sum(y)))
    negatives = max(1.0, float(len(y) - np.sum(y)))
    pos_weight = negatives / positives
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, 128):
            idx = order[start : start + 128]
            xb = x[idx]
            yb = y[idx]
            logits = xb @ weights + bias
            pred = 1.0 / (1.0 + np.exp(-logits))
            sample_weight = np.where(yb >= 0.5, pos_weight, 1.0)
            err = (pred - yb) * sample_weight
            weights -= lr * ((xb.T @ err) / len(idx) + l2 * weights)
            bias -= lr * float(np.mean(err))
    return weights, bias


def predict_prob(x, weights, bias):
    return np.asarray([sigmoid(float(row @ weights + bias)) for row in x])


def accuracy(x, y, weights, bias, threshold):
    pred = predict_prob(x, weights, bias) >= threshold
    return float(np.mean(pred == (y >= 0.5)))


def balanced_accuracy_from_probs(probs, y, threshold):
    pred = probs >= threshold
    truth = y >= 0.5
    pos = truth
    neg = ~truth
    tpr = float(np.mean(pred[pos] == truth[pos])) if np.any(pos) else 1.0
    tnr = float(np.mean(pred[neg] == truth[neg])) if np.any(neg) else 1.0
    return 0.5 * (tpr + tnr)


def tune_threshold(x, y, weights, bias):
    probs = predict_prob(x, weights, bias)
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        score = balanced_accuracy_from_probs(probs, y, float(threshold))
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, best_score


def main():
    parser = argparse.ArgumentParser(description="Train a Stage 1 flee model.")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed-start", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.0005)
    parser.add_argument("--out", default="flee_model.json")
    args = parser.parse_args()

    random.seed(args.seed_start)
    x, y = collect_examples(args.samples, args.seed_start)
    split = max(1, int(len(y) * 0.8))
    weights, bias = train_logistic(x[:split], y[:split], args.epochs, args.lr, args.l2)
    if split < len(y):
        threshold, valid_balanced_accuracy = tune_threshold(x[split:], y[split:], weights, bias)
    else:
        threshold, valid_balanced_accuracy = tune_threshold(x[:split], y[:split], weights, bias)
    train_acc = accuracy(x[:split], y[:split], weights, bias, threshold)
    valid_acc = accuracy(x[split:], y[split:], weights, bias, threshold) if split < len(y) else train_acc
    payload = {
        "type": "logistic_flee_policy",
        "observation_size": observation_size(),
        "weights": weights.astype(float).tolist(),
        "bias": float(bias),
        "threshold": threshold,
        "samples": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "train_accuracy": train_acc,
        "valid_accuracy": valid_acc,
        "valid_balanced_accuracy": valid_balanced_accuracy,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(
        f"wrote {args.out}: samples={len(y)} positive={np.mean(y):.3f} "
        f"threshold={threshold:.2f} train_acc={train_acc:.3f} valid_acc={valid_acc:.3f} "
        f"valid_bal_acc={valid_balanced_accuracy:.3f}"
    )


if __name__ == "__main__":
    main()
