import json
import random

import numpy as np

from .flee_features import extract_flee_observation, observation_size, sigmoid
from .replay_features import extract_replay_observation, observation_size as replay_observation_size
from .break_features import MAX_OBJECTS, extract_break_observation, observation_size as break_observation_size
from .scry_features import extract_scry_observation, observation_size as scry_observation_size
from .joint_features import (
    DECISION_ITEM_ACTIVATION,
    DECISION_SCRY_WINDOW,
    extract_joint_action_observation,
    observation_size as joint_observation_size,
)
from .item_features import (
    DECK_MANIPULATION_ITEM_CLASSES,
    PEEK_COMBO_ITEM_CLASSES,
    SCRY_ITEM_CLASSES,
    extract_item_activation_observation,
    extract_item_activation_observation_legacy,
    extract_item_activation_observation_v2,
    extract_item_activation_observation_v3,
    legacy_observation_size as legacy_item_observation_size,
    observation_size as item_observation_size,
    v2_observation_size as v2_item_observation_size,
    v3_observation_size as v3_item_observation_size,
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

    def choose_scry_window_action(self, state, legal_actions):
        player = state["player"]
        cards = state.get("cards", [])
        mask = 0
        for idx, card in enumerate(cards[:2]):
            if hasattr(card, "types") and not getattr(card, "event", False) and card.puissance >= player.pv_total:
                mask |= 1 << idx
        return mask if mask in legal_actions else (legal_actions[0] if legal_actions else 0)


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

    def choose_scry_window_action(self, state, legal_actions):
        player = state["player"]
        cards = state.get("cards", [])
        mask = 0
        for idx, card in enumerate(cards[:2]):
            if hasattr(card, "types") and not getattr(card, "event", False) and card.puissance >= player.pv_total:
                mask |= 1 << idx
        return mask if mask in legal_actions else (legal_actions[0] if legal_actions else 0)


class CombinedPolicy(GamePolicy):
    def __init__(self, flee_policy=None, replay_policy=None, break_policy=None, item_policy=None, scry_policy=None):
        self.flee_policy = flee_policy or HeuristicPolicy("ev")
        self.replay_policy = replay_policy or HeuristicPolicy("ev")
        self.break_policy = break_policy or HeuristicPolicy("ev")
        self.item_policy = item_policy or HeuristicPolicy("ev")
        self.scry_policy = scry_policy or self.item_policy

    def decide_flee(self, state, legal_actions):
        return self.flee_policy.decide_flee(state, legal_actions)

    def decide_replay(self, state, legal_actions):
        return self.replay_policy.decide_replay(state, legal_actions)

    def choose_item_to_break(self, state, legal_actions):
        return self.break_policy.choose_item_to_break(state, legal_actions)

    def choose_item_activation(self, state, legal_actions):
        return self.item_policy.choose_item_activation(state, legal_actions)

    def choose_scry_window_action(self, state, legal_actions):
        return self.scry_policy.choose_scry_window_action(state, legal_actions)


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

    def choose_scry_window_action(self, state, legal_actions):
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

    def choose_scry_window_action(self, state, legal_actions):
        return self.fallback.choose_scry_window_action(state, legal_actions)


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

    def choose_scry_window_action(self, state, legal_actions):
        return HeuristicPolicy("ev").choose_scry_window_action(state, legal_actions)


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


def _known_card_flee_action(state, legal_actions):
    if FLEE_ACTION_ATTEMPT not in legal_actions:
        return None
    joueur = state["player"]
    jeu = state["game"]
    known = joueur.connait_prochaine_carte(jeu)
    if known is None or getattr(known, "is_X", False):
        return None
    if getattr(known, "event", False):
        return FLEE_ACTION_CONTINUE
    if joueur.peut_executer_facilement(known) or getattr(known, "puissance", 0) <= 2:
        return FLEE_ACTION_CONTINUE
    if joueur._degats_attendus(known, jeu) >= joueur.pv_total and joueur._nb_options_combat() <= 1:
        return FLEE_ACTION_ATTEMPT
    return None


class KnownCardGuardFleePolicy(GamePolicy):
    def __init__(self, base_policy):
        self.base_policy = base_policy

    def decide_flee(self, state, legal_actions):
        guarded = _known_card_flee_action(state, legal_actions)
        if guarded is not None:
            return guarded
        return self.base_policy.decide_flee(state, legal_actions)

    def decide_replay(self, state, legal_actions):
        return self.base_policy.decide_replay(state, legal_actions)

    def choose_item_to_break(self, state, legal_actions):
        return self.base_policy.choose_item_to_break(state, legal_actions)

    def choose_item_activation(self, state, legal_actions):
        return self.base_policy.choose_item_activation(state, legal_actions)

    def choose_scry_window_action(self, state, legal_actions):
        return self.base_policy.choose_scry_window_action(state, legal_actions)


class KnownCardGuardReplayPolicy(GamePolicy):
    def __init__(self, base_policy):
        self.base_policy = base_policy
        self.heuristic = HeuristicPolicy("ev")

    def decide_flee(self, state, legal_actions):
        return self.base_policy.decide_flee(state, legal_actions)

    def decide_replay(self, state, legal_actions):
        if state["player"].connait_prochaine_carte(state["game"]) is not None:
            return self.heuristic.decide_replay(state, legal_actions)
        return self.base_policy.decide_replay(state, legal_actions)

    def choose_item_to_break(self, state, legal_actions):
        return self.base_policy.choose_item_to_break(state, legal_actions)

    def choose_item_activation(self, state, legal_actions):
        return self.base_policy.choose_item_activation(state, legal_actions)

    def choose_scry_window_action(self, state, legal_actions):
        return self.base_policy.choose_scry_window_action(state, legal_actions)


class NumpyReplayPolicy(NumpyPPOFleePolicy):
    def __init__(self, model_path, flee_policy=None):
        with open(model_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        self.model_type = payload.get("type", "sb3_ppo_actor_tanh")
        self.layers = [
            (
                np.asarray(layer["weight"], dtype=np.float32),
                np.asarray(layer["bias"], dtype=np.float32),
            )
            for layer in payload["policy_layers"]
        ]
        self.action_weight = np.asarray(payload.get("action_weight", payload.get("q_weight")), dtype=np.float32)
        self.action_bias = np.asarray(payload.get("action_bias", payload.get("q_bias")), dtype=np.float32)
        self.q_draw_bias = float(payload.get("metadata", {}).get("q_draw_bias", 0.0))
        self.flee_policy = flee_policy or HeuristicPolicy("ev")
        expected_input = replay_observation_size() + 2 if self.model_type == "replay_q_tanh" else replay_observation_size()
        if self.layers[0][0].shape[1] != expected_input:
            raise ValueError(
                f"model expects {self.layers[0][0].shape[1]} features, "
                f"got {expected_input}"
            )

    def decide_flee(self, state, legal_actions):
        return self.flee_policy.decide_flee(state, legal_actions)

    def decide_replay(self, state, legal_actions):
        if REPLAY_ACTION_DRAW not in legal_actions:
            return REPLAY_ACTION_PASS
        x = extract_replay_observation(state["player"], state["game"])
        if self.model_type == "replay_q_tanh":
            values = {}
            for action in legal_actions:
                if int(action) not in (REPLAY_ACTION_PASS, REPLAY_ACTION_DRAW):
                    continue
                action_features = np.asarray([1.0 if int(action) == 0 else 0.0, 1.0 if int(action) == 1 else 0.0])
                qx = np.concatenate([x, action_features]).astype(np.float32)
                for weight, bias in self.layers:
                    qx = np.tanh(weight @ qx + bias)
                value = float((self.action_weight @ qx + self.action_bias).reshape(-1)[0])
                if int(action) == REPLAY_ACTION_DRAW:
                    value += self.q_draw_bias
                values[int(action)] = value
            if values:
                return max(values, key=values.get)
            return REPLAY_ACTION_PASS
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
        self.model_type = payload.get("type", "item_activation_actor_tanh")
        self.layers = [
            (
                np.asarray(layer["weight"], dtype=np.float32),
                np.asarray(layer["bias"], dtype=np.float32),
            )
            for layer in payload["policy_layers"]
        ]
        self.action_weight = np.asarray(payload.get("action_weight", payload.get("q_weight")), dtype=np.float32)
        self.action_bias = np.asarray(payload.get("action_bias", payload.get("q_bias")), dtype=np.float32)
        metadata = payload.get("metadata", {})
        self.force_survival = bool(metadata.get("force_survival", self.model_type == "item_activation_q_tanh"))
        self.q_use_bias = float(metadata.get("q_use_bias", 0.0))
        self.item_hook_specialists = {}
        for key, specialist in metadata.get("item_hook_specialists", {}).items():
            self.item_hook_specialists[key] = {
                "layers": [
                    (
                        np.asarray(layer["weight"], dtype=np.float32),
                        np.asarray(layer["bias"], dtype=np.float32),
                    )
                    for layer in specialist["policy_layers"]
                ],
                "q_weight": np.asarray(specialist["q_weight"], dtype=np.float32),
                "q_bias": np.asarray(specialist["q_bias"], dtype=np.float32),
            }
        self.flee_policy = flee_policy or HeuristicPolicy("ev")
        self.replay_policy = replay_policy or HeuristicPolicy("ev")
        self.break_policy = break_policy or HeuristicPolicy("ev")
        self.input_size = int(payload.get("observation_size", self.layers[0][0].shape[1]))
        if self.model_type == "item_activation_q_tanh":
            self.network_input_size = self.layers[0][0].shape[1]
            if self.network_input_size != self.input_size + 2:
                raise ValueError(
                    f"Q item model expects network input {self.network_input_size}, "
                    f"but observation_size is {self.input_size}"
                )
        if self.input_size == item_observation_size():
            self.extract_observation = extract_item_activation_observation
        elif self.input_size == v3_item_observation_size():
            self.extract_observation = extract_item_activation_observation_v3
        elif self.input_size == v2_item_observation_size():
            self.extract_observation = extract_item_activation_observation_v2
        elif self.input_size == legacy_item_observation_size():
            self.extract_observation = extract_item_activation_observation_legacy
        else:
            raise ValueError(
                f"model expects {self.input_size} features, "
                f"got supported sizes {legacy_item_observation_size()}, "
                f"{v2_item_observation_size()}, {v3_item_observation_size()}, "
                f"or {item_observation_size()}"
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
        if self.force_survival and state.get("hook", "") == "en_survie":
            return 1
        x = self.extract_observation(
            state["player"], state["game"], state["item"], state["card"], state.get("hook", "")
        )
        if self.model_type == "item_activation_q_tanh":
            item_class = type(state["item"]).__name__
            hook = state.get("hook", "")
            specialist = self.item_hook_specialists.get(f"{item_class}|{hook}")
            values = {}
            for action in legal_actions:
                if int(action) not in (0, 1):
                    continue
                action_features = np.asarray([1.0 if int(action) == 0 else 0.0, 1.0 if int(action) == 1 else 0.0])
                qx = np.concatenate([x, action_features]).astype(np.float32)
                layers = specialist["layers"] if specialist is not None else self.layers
                action_weight = specialist["q_weight"] if specialist is not None else self.action_weight
                action_bias = specialist["q_bias"] if specialist is not None else self.action_bias
                for weight, bias in layers:
                    qx = np.tanh(weight @ qx + bias)
                value = float((action_weight @ qx + action_bias).reshape(-1)[0])
                if int(action) == 1:
                    value += self.q_use_bias
                values[int(action)] = value
            if values:
                return max(values, key=values.get)
            return 0
        for weight, bias in self.layers:
            x = np.tanh(weight @ x + bias)
        logits = self.action_weight @ x + self.action_bias
        action = int(np.argmax(logits))
        return action if action in legal_actions else 0


class HybridScryItemActivationPolicy(GamePolicy):
    def __init__(self, base_model_path, scry_model_path, flee_policy=None, replay_policy=None, break_policy=None):
        self.base_policy = NumpyItemActivationPolicy(
            base_model_path,
            flee_policy=flee_policy,
            replay_policy=replay_policy,
            break_policy=break_policy,
        )
        self.scry_policy = NumpyItemActivationPolicy(
            scry_model_path,
            flee_policy=flee_policy,
            replay_policy=replay_policy,
            break_policy=break_policy,
        )
        self.scry_item_classes = SCRY_ITEM_CLASSES | PEEK_COMBO_ITEM_CLASSES | DECK_MANIPULATION_ITEM_CLASSES

    def decide_flee(self, state, legal_actions):
        return self.base_policy.decide_flee(state, legal_actions)

    def decide_replay(self, state, legal_actions):
        return self.base_policy.decide_replay(state, legal_actions)

    def choose_item_to_break(self, state, legal_actions):
        return self.base_policy.choose_item_to_break(state, legal_actions)

    def choose_item_activation(self, state, legal_actions):
        item_class = type(state["item"]).__name__
        if item_class in self.scry_item_classes:
            return self.scry_policy.choose_item_activation(state, legal_actions)
        return self.base_policy.choose_item_activation(state, legal_actions)


class NumpyScryWindowPolicy(GamePolicy):
    def __init__(self, model_path, fallback=None):
        with open(model_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        self.layers = [
            (
                np.asarray(layer["weight"], dtype=np.float32),
                np.asarray(layer["bias"], dtype=np.float32),
            )
            for layer in payload["policy_layers"]
        ]
        self.q_weight = np.asarray(payload["q_weight"], dtype=np.float32)
        self.q_bias = np.asarray(payload["q_bias"], dtype=np.float32)
        self.input_size = int(payload.get("observation_size", self.layers[0][0].shape[1]))
        if self.input_size != scry_observation_size():
            raise ValueError(f"scry model expects {self.input_size}, supported {scry_observation_size()}")
        self.fallback_margin = float(payload.get("metadata", {}).get("fallback_margin", -1.0e9))
        self.fallback = fallback or HeuristicPolicy("ev")

    def decide_flee(self, state, legal_actions):
        return self.fallback.decide_flee(state, legal_actions)

    def decide_replay(self, state, legal_actions):
        return self.fallback.decide_replay(state, legal_actions)

    def choose_item_to_break(self, state, legal_actions):
        return self.fallback.choose_item_to_break(state, legal_actions)

    def choose_item_activation(self, state, legal_actions):
        return self.fallback.choose_item_activation(state, legal_actions)

    def choose_scry_window_action(self, state, legal_actions):
        if not legal_actions:
            return 0
        x = extract_scry_observation(state["player"], state["game"], state.get("cards", []), state.get("source", ""))
        for weight, bias in self.layers:
            x = np.tanh(weight @ x + bias)
        logits = self.q_weight @ x + self.q_bias
        masked = np.full(4, -1.0e9, dtype=np.float32)
        for action in legal_actions:
            if 0 <= int(action) < 4:
                masked[int(action)] = logits[int(action)]
        model_action = int(np.argmax(masked))
        if self.fallback_margin > -1.0e8:
            fallback_action = int(self.fallback.choose_scry_window_action(state, legal_actions))
            if 0 <= fallback_action < 4 and masked[model_action] - masked[fallback_action] < self.fallback_margin:
                return fallback_action
        return model_action


class NumpyJointDecisionPolicy(GamePolicy):
    def __init__(self, model_path, fallback=None):
        with open(model_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        self.layers = [
            (
                np.asarray(layer["weight"], dtype=np.float32),
                np.asarray(layer["bias"], dtype=np.float32),
            )
            for layer in payload["policy_layers"]
        ]
        self.q_weight = np.asarray(payload["q_weight"], dtype=np.float32)
        self.q_bias = np.asarray(payload["q_bias"], dtype=np.float32)
        self.input_size = int(payload.get("observation_size", self.layers[0][0].shape[1]))
        if self.input_size != joint_observation_size():
            raise ValueError(f"joint model expects {self.input_size}, supported {joint_observation_size()}")
        self.force_survival = bool(payload.get("metadata", {}).get("force_survival", True))
        self.action_bias_by_kind = {
            str(kind): {int(action): float(value) for action, value in actions.items()}
            for kind, actions in payload.get("metadata", {}).get("action_bias_by_kind", {}).items()
        }
        self.fallback = fallback or HeuristicPolicy("ev")

    def decide_flee(self, state, legal_actions):
        return self.fallback.decide_flee(state, legal_actions)

    def decide_replay(self, state, legal_actions):
        return self.fallback.decide_replay(state, legal_actions)

    def choose_item_to_break(self, state, legal_actions):
        return self.fallback.choose_item_to_break(state, legal_actions)

    def _score(self, state, kind, legal_actions):
        values = {}
        for action in legal_actions:
            action = int(action)
            x = extract_joint_action_observation(state, kind, action)
            for weight, bias in self.layers:
                x = np.tanh(weight @ x + bias)
            values[action] = float((self.q_weight @ x + self.q_bias).reshape(-1)[0])
            values[action] += self.action_bias_by_kind.get(kind, {}).get(action, 0.0)
        if not values:
            return 0
        return max(values, key=values.get)

    def choose_item_activation(self, state, legal_actions):
        if 1 not in legal_actions:
            return legal_actions[0] if legal_actions else 0
        if self.force_survival and state.get("hook", "") == "en_survie":
            return 1
        return self._score(state, DECISION_ITEM_ACTIVATION, legal_actions)

    def choose_scry_window_action(self, state, legal_actions):
        if not legal_actions:
            return 0
        return self._score(state, DECISION_SCRY_WINDOW, legal_actions)


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
