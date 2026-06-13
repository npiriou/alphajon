import hashlib

import numpy as np

from .break_features import extract_break_observation, observation_size as break_observation_size

ITEM_BUCKETS = 64
ITEM_EXTRA_FEATURES = 15


def _bucket(text):
    raw = hashlib.sha1(str(text).encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(raw[:4], "little") % ITEM_BUCKETS


def extract_item_activation_observation(joueur, jeu, item, card, hook):
    base = extract_break_observation(joueur, jeu).tolist()
    item_index = joueur.objets.index(item) if item in joueur.objets else -1
    item_one_hot = [0.0] * ITEM_BUCKETS
    item_one_hot[_bucket(getattr(item, "nom", type(item).__name__))] = 1.0
    card_power = getattr(card, "puissance", getattr(card, "puissance_initiale", 0))
    card_damage = getattr(card, "dommages", card_power)
    card_is_event = 1.0 if getattr(card, "event", False) else 0.0
    card_is_x = 1.0 if getattr(card, "is_X", False) else 0.0
    item_features = [
        item_index / 12.0,
        1.0 if getattr(item, "intact", False) else 0.0,
        1.0 if getattr(item, "actif", False) else 0.0,
        getattr(item, "pv_bonus", 0) / 10.0,
        getattr(item, "modificateur_de", 0) / 10.0,
        getattr(item, "priorite", 0) / 100.0,
        len(getattr(item, "types_tags", ())) / 8.0,
        len(getattr(item, "puissance_tags", ())) / 10.0,
        1.0 if getattr(item, "pv_bonus", 0) >= joueur.pv_total else 0.0,
        card_power / 10.0,
        card_damage / 10.0,
        1.0 if card_damage >= joueur.pv_total else 0.0,
        card_is_event,
        card_is_x,
        1.0 if hook == "en_combat" else 0.0,
    ]
    return np.asarray(base + item_features + item_one_hot, dtype=np.float32)


def observation_size():
    return break_observation_size() + ITEM_EXTRA_FEATURES + ITEM_BUCKETS
