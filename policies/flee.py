import json
import random

import numpy as np

from .flee_features import extract_flee_observation, observation_size, sigmoid
from .replay_features import extract_replay_observation, observation_size as replay_observation_size
from .break_features import MAX_OBJECTS, extract_break_observation, observation_size as break_observation_size
from .item_features import (
    extract_item_activation_observation,
    extract_item_activation_observation_legacy,
    legacy_observation_size as legacy_item_observation_size,
    observation_size as item_observation_size,
)

FLEE_ACTION_CONTINUE = 0
FLEE_ACTION_ATTEMPT = 1
REPLAY_ACTION_PASS = 0
REPLAY_ACTION_DRAW = 1


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

    def decide_replay(self, state, legal_actions):
        joueur = state["player"]
        jeu = state["game"]
        return REPLAY_ACTION_DRAW if joueur._decision_replay_heuristic(jeu, state.get("log_details", [])) else REPLAY_ACTION_PASS

    def choose_item_to_break(self, state, legal_actions):
        objet = state["player"]._choose_break_object_heuristic(state["game"])
        if objet is None:
            return legal_actions[0] if legal_actions else 0
        return state["player"].objets.index(objet)

    def choose_item_activation(self, state, legal_actions):
        if 1 not in legal_actions:
            return legal_actions[0] if legal_actions else 0
        hook = state.get("hook", "")
        if hook == "en_survie":
            return 1
        if hook == "en_combat":
            item = state["item"]
            return 1 if item.worthit(
                state["player"],
                state["card"],
                state["game"],
                state.get("log_details", []),
            ) else 0
        return 0


class CombinedPolicy(GamePolicy):
    def __init__(self, flee_policy=None, replay_policy=None, break_policy=None, item_policy=None):
        self.flee_policy = flee_policy or HeuristicPolicy("ev")
        self.replay_policy = replay_policy or HeuristicPolicy("ev")
        self.break_policy = break_policy or HeuristicPolicy("ev")
        self.item_policy = item_policy or HeuristicPolicy("ev")

    def decide_flee(self, state, legal_actions):
        return self.flee_policy.decide_flee(state, legal_actions)

    def decide_replay(self, state, legal_actions):
        return self.replay_policy.decide_replay(state, legal_actions)

    def choose_item_to_break(self, state, legal_actions):
        return self.break_policy.choose_item_to_break(state, legal_actions)

    def choose_item_activation(self, state, legal_actions):
        return self.item_policy.choose_item_activation(state, legal_actions)


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

    def decide_replay(self, state, legal_actions):
        if REPLAY_ACTION_DRAW not in legal_actions:
            return REPLAY_ACTION_PASS
        return REPLAY_ACTION_DRAW if self.rng.random() < 0.5 else REPLAY_ACTION_PASS

    def choose_item_to_break(self, state, legal_actions):
        return self.rng.choice(list(legal_actions)) if legal_actions else 0

    def choose_item_activation(self, state, legal_actions):
        return self.rng.choice(list(legal_actions)) if legal_actions else 0


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

    def decide_replay(self, state, legal_actions):
        return self.fallback.decide_replay(state, legal_actions)

    def choose_item_to_break(self, state, legal_actions):
        return self.fallback.choose_item_to_break(state, legal_actions)

    def choose_item_activation(self, state, legal_actions):
        return self.fallback.choose_item_activation(state, legal_actions)


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

    def decide_replay(self, state, legal_actions):
        return HeuristicPolicy("ev").decide_replay(state, legal_actions)

    def choose_item_to_break(self, state, legal_actions):
        return HeuristicPolicy("ev").choose_item_to_break(state, legal_actions)

    def choose_item_activation(self, state, legal_actions):
        return HeuristicPolicy("ev").choose_item_activation(state, legal_actions)


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

    def decide_replay(self, state, legal_actions):
        return HeuristicPolicy("ev").decide_replay(state, legal_actions)

    def choose_item_to_break(self, state, legal_actions):
        return HeuristicPolicy("ev").choose_item_to_break(state, legal_actions)

    def choose_item_activation(self, state, legal_actions):
        return HeuristicPolicy("ev").choose_item_activation(state, legal_actions)


class NumpyReplayPolicy(NumpyPPOFleePolicy):
    def __init__(self, model_path, flee_policy=None):
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
        self.flee_policy = flee_policy or HeuristicPolicy("ev")
        if self.layers[0][0].shape[1] != replay_observation_size():
            raise ValueError(
                f"model expects {self.layers[0][0].shape[1]} features, "
                f"got {replay_observation_size()}"
            )

    def decide_flee(self, state, legal_actions):
        return self.flee_policy.decide_flee(state, legal_actions)

    def decide_replay(self, state, legal_actions):
        if REPLAY_ACTION_DRAW not in legal_actions:
            return REPLAY_ACTION_PASS
        x = extract_replay_observation(state["player"], state["game"])
        for weight, bias in self.layers:
            x = np.tanh(weight @ x + bias)
        logits = self.action_weight @ x + self.action_bias
        action = int(np.argmax(logits))
        return action if action in legal_actions else REPLAY_ACTION_PASS

    def choose_item_to_break(self, state, legal_actions):
        return HeuristicPolicy("ev").choose_item_to_break(state, legal_actions)

    def choose_item_activation(self, state, legal_actions):
        return HeuristicPolicy("ev").choose_item_activation(state, legal_actions)


class NumpyBreakPolicy(GamePolicy):
    def __init__(self, model_path, flee_policy=None, replay_policy=None):
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
        self.flee_policy = flee_policy or HeuristicPolicy("ev")
        self.replay_policy = replay_policy or HeuristicPolicy("ev")
        if self.layers[0][0].shape[1] != break_observation_size():
            raise ValueError(
                f"model expects {self.layers[0][0].shape[1]} features, "
                f"got {break_observation_size()}"
            )

    def decide_flee(self, state, legal_actions):
        return self.flee_policy.decide_flee(state, legal_actions)

    def decide_replay(self, state, legal_actions):
        return self.replay_policy.decide_replay(state, legal_actions)

    def choose_item_to_break(self, state, legal_actions):
        if not legal_actions:
            return 0
        x = extract_break_observation(state["player"], state["game"])
        for weight, bias in self.layers:
            x = np.tanh(weight @ x + bias)
        logits = self.action_weight @ x + self.action_bias
        masked = np.full(MAX_OBJECTS, -1.0e9, dtype=np.float32)
        for action in legal_actions:
            if 0 <= int(action) < MAX_OBJECTS:
                masked[int(action)] = logits[int(action)]
        return int(np.argmax(masked))

    def choose_item_activation(self, state, legal_actions):
        return HeuristicPolicy("ev").choose_item_activation(state, legal_actions)


class NumpyItemActivationPolicy(GamePolicy):
    def __init__(self, model_path, flee_policy=None, replay_policy=None, break_policy=None):
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
        self.flee_policy = flee_policy or HeuristicPolicy("ev")
        self.replay_policy = replay_policy or HeuristicPolicy("ev")
        self.break_policy = break_policy or HeuristicPolicy("ev")
        self.input_size = self.layers[0][0].shape[1]
        if self.input_size == item_observation_size():
            self.extract_observation = extract_item_activation_observation
        elif self.input_size == legacy_item_observation_size():
            self.extract_observation = extract_item_activation_observation_legacy
        else:
            raise ValueError(
                f"model expects {self.input_size} features, "
                f"got supported sizes {legacy_item_observation_size()} or {item_observation_size()}"
            )

    def decide_flee(self, state, legal_actions):
        return self.flee_policy.decide_flee(state, legal_actions)

    def decide_replay(self, state, legal_actions):
        return self.replay_policy.decide_replay(state, legal_actions)

    def choose_item_to_break(self, state, legal_actions):
        return self.break_policy.choose_item_to_break(state, legal_actions)

    def choose_item_activation(self, state, legal_actions):
        if 1 not in legal_actions:
            return legal_actions[0] if legal_actions else 0
        x = self.extract_observation(
            state["player"], state["game"], state["item"], state["card"], state.get("hook", "")
        )
        for weight, bias in self.layers:
            x = np.tanh(weight @ x + bias)
        logits = self.action_weight @ x + self.action_bias
        action = int(np.argmax(logits))
        return action if action in legal_actions else 0


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

    def decide_replay(self, state, legal_actions):
        return HeuristicPolicy("ev").decide_replay(state, legal_actions)

    def choose_item_to_break(self, state, legal_actions):
        return HeuristicPolicy("ev").choose_item_to_break(state, legal_actions)

    def choose_item_activation(self, state, legal_actions):
        return HeuristicPolicy("ev").choose_item_activation(state, legal_actions)
