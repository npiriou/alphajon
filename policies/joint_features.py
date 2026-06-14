import numpy as np

from .item_features import extract_item_activation_observation, observation_size as item_observation_size
from .scry_features import extract_scry_observation, observation_size as scry_observation_size

FEATURE_VERSION = "joint_decision_q_v1"
DECISION_ITEM_ACTIVATION = "item_activation"
DECISION_SCRY_WINDOW = "scry_window"
ACTION_FEATURES = 16
STATE_FEATURES = max(item_observation_size(), scry_observation_size())


def _padded_state(obs):
    padded = np.zeros(STATE_FEATURES, dtype=np.float32)
    padded[: len(obs)] = obs
    return padded


def _action_features(kind, action):
    action = int(action)
    features = np.zeros(ACTION_FEATURES, dtype=np.float32)
    if 0 <= action < 4:
        features[action] = 1.0
    features[4] = float(action) / 3.0
    if kind == DECISION_ITEM_ACTIVATION:
        features[5] = 1.0 if action == 1 else 0.0
        features[6] = 1.0 if action == 0 else 0.0
    elif kind == DECISION_SCRY_WINDOW:
        features[7] = 1.0 if action & 1 else 0.0
        features[8] = 1.0 if action & 2 else 0.0
        features[9] = float((action & 1) + ((action >> 1) & 1)) / 2.0
        features[10] = 1.0 if action == 0 else 0.0
        features[11] = 1.0 if action == 3 else 0.0
    return features


def extract_joint_state_observation(state, kind):
    if kind == DECISION_ITEM_ACTIVATION:
        obs = extract_item_activation_observation(
            state["player"],
            state["game"],
            state["item"],
            state["card"],
            state.get("hook", ""),
        )
        kind_features = np.asarray([1.0, 0.0], dtype=np.float32)
    elif kind == DECISION_SCRY_WINDOW:
        obs = extract_scry_observation(
            state["player"],
            state["game"],
            state.get("cards", []),
            state.get("source", ""),
        )
        kind_features = np.asarray([0.0, 1.0], dtype=np.float32)
    else:
        raise ValueError(f"unknown joint decision kind: {kind}")
    return np.concatenate([kind_features, _padded_state(obs)]).astype(np.float32)


def extract_joint_action_observation(state, kind, action):
    base = extract_joint_state_observation(state, kind)
    return np.concatenate([base, _action_features(kind, action)]).astype(np.float32)


def observation_size():
    return 2 + STATE_FEATURES + ACTION_FEATURES


def state_observation_size():
    return 2 + STATE_FEATURES
