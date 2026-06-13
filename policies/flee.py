import json
import random

import numpy as np

from .flee_features import extract_flee_observation, observation_size, sigmoid

FLEE_ACTION_CONTINUE = 0
FLEE_ACTION_ATTEMPT = 1


class GamePolicy:
    def decide_flee(self, state, legal_actions):
        raise NotImplementedError

    def decide_replay(self, state, legal_actions):
        raise NotImplementedError

    def choose_item_to_break(self, state, legal_actions):
        raise NotImplementedError

    def choose_draft_pick(self, state, legal_items):
        raise NotImplementedError

    def choose_item_activation(self, state, legal_actions):
        raise NotImplementedError


class HeuristicPolicy(GamePolicy):
    """Wraps the current Joueur flee behavior."""

    def __init__(self, mode=None):
        self.mode = mode

    def decide_flee(self, state, legal_actions):
        joueur = state["player"]
        jeu = state["game"]
        mode = self.mode or getattr(joueur, "politique_fuite", "ev")
        if mode == "ev":
            return FLEE_ACTION_ATTEMPT if joueur._decision_fuite_ev(jeu) else FLEE_ACTION_CONTINUE
        return FLEE_ACTION_ATTEMPT if joueur._decision_fuite_seuils(jeu) else FLEE_ACTION_CONTINUE


class RandomPolicy(GamePolicy):
    def __init__(self, attempt_probability=0.5, rng=None):
        self.attempt_probability = attempt_probability
        self.rng = rng or random

    def decide_flee(self, state, legal_actions):
        if FLEE_ACTION_ATTEMPT not in legal_actions:
            return FLEE_ACTION_CONTINUE
        return (
            FLEE_ACTION_ATTEMPT
            if self.rng.random() < self.attempt_probability
            else FLEE_ACTION_CONTINUE
        )


class ScriptedPolicy(GamePolicy):
    def __init__(self, actions, fallback=None):
        self.actions = list(actions)
        self.index = 0
        self.fallback = fallback or HeuristicPolicy()

    def decide_flee(self, state, legal_actions):
        if self.index < len(self.actions):
            action = int(self.actions[self.index])
            self.index += 1
            if action in legal_actions:
                return action
            return FLEE_ACTION_CONTINUE
        return self.fallback.decide_flee(state, legal_actions)


class ModelPolicy(GamePolicy):
    """Small logistic flee policy loaded from JSON.

    This is intentionally lightweight so Stage 1 can train/evaluate without a
    heavy RL stack. PPO policies can later implement the same GamePolicy method.
    """

    def __init__(self, model_path, threshold=0.5):
        with open(model_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        self.weights = np.asarray(payload["weights"], dtype=np.float32)
        self.bias = float(payload.get("bias", 0.0))
        self.threshold = float(payload.get("threshold", threshold))
        if self.weights.shape[0] != observation_size():
            raise ValueError(
                f"model has {self.weights.shape[0]} weights, expected {observation_size()}"
            )

    def decide_flee(self, state, legal_actions):
        if FLEE_ACTION_ATTEMPT not in legal_actions:
            return FLEE_ACTION_CONTINUE
        obs = extract_flee_observation(state["player"], state["game"])
        p = sigmoid(float(obs @ self.weights + self.bias))
        return FLEE_ACTION_ATTEMPT if p >= self.threshold else FLEE_ACTION_CONTINUE


class NumpyPPOFleePolicy(GamePolicy):
    def __init__(self, model_path):
        with open(model_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        self.layers = [
            (
                np.asarray(layer["weight"], dtype=np.float32),
                np.asarray(layer["bias"], dtype=np.float32),
            )
            for layer in payload["policy_layers"]
        ]
        self.action_weight = np.asarray(payload["action_weight"], dtype=np.float32)
        self.action_bias = np.asarray(payload["action_bias"], dtype=np.float32)
        if self.layers[0][0].shape[1] != observation_size():
            raise ValueError(
                f"model expects {self.layers[0][0].shape[1]} features, "
                f"got {observation_size()}"
            )

    def decide_flee(self, state, legal_actions):
        if FLEE_ACTION_ATTEMPT not in legal_actions:
            return FLEE_ACTION_CONTINUE
        x = extract_flee_observation(state["player"], state["game"])
        for weight, bias in self.layers:
            x = np.tanh(weight @ x + bias)
        logits = self.action_weight @ x + self.action_bias
        action = int(np.argmax(logits))
        return action if action in legal_actions else FLEE_ACTION_CONTINUE


class StableBaselinesFleePolicy(GamePolicy):
    def __init__(self, model_path, deterministic=True):
        from stable_baselines3 import PPO

        self.model = PPO.load(model_path, device="cpu")
        self.deterministic = deterministic

    def decide_flee(self, state, legal_actions):
        if FLEE_ACTION_ATTEMPT not in legal_actions:
            return FLEE_ACTION_CONTINUE
        obs = extract_flee_observation(state["player"], state["game"])
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        action = int(action)
        return action if action in legal_actions else FLEE_ACTION_CONTINUE
