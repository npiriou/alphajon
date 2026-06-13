import hashlib

import numpy as np

from .break_features import extract_break_observation, observation_size as break_observation_size

LEGACY_ITEM_BUCKETS = 64
ITEM_CLASS_BUCKETS = 320
HOOK_BUCKETS = 8
ITEM_EXTRA_FEATURES = 19
LEGACY_ITEM_EXTRA_FEATURES = 15
FEATURE_VERSION = "item_activation_v2"
LEGACY_FEATURE_VERSION = "item_activation_legacy"
HOOK_ORDER = {
    "en_combat": 0,
    "en_survie": 1,
    "en_fuite": 2,
    "debut_tour": 3,
    "fin_tour": 4,
    "en_vaincu": 5,
    "en_subit_dommages": 6,
}


def _item_class_index(item):
    from objets import objets_disponibles

    cls = type(item)
    seen = []
    for obj in objets_disponibles:
        obj_cls = type(obj)
        if obj_cls not in seen:
            seen.append(obj_cls)
    try:
        return seen.index(cls)
    except ValueError:
        return -1


def _bucket(text, buckets):
    raw = hashlib.sha1(str(text).encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(raw[:4], "little") % buckets


def _common_item_features(joueur, item, card, hook):
    item_index = joueur.objets.index(item) if item in joueur.objets else -1
    card_power = getattr(card, "puissance", getattr(card, "puissance_initiale", 0))
    card_damage = getattr(card, "dommages", card_power)
    return item_index, card_power, card_damage, [
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
        1.0 if getattr(card, "event", False) else 0.0,
        1.0 if getattr(card, "is_X", False) else 0.0,
        1.0 if hook == "en_combat" else 0.0,
    ]


def extract_item_activation_observation_legacy(joueur, jeu, item, card, hook):
    base = extract_break_observation(joueur, jeu).tolist()
    _, _, _, item_features = _common_item_features(joueur, item, card, hook)
    item_one_hot = [0.0] * LEGACY_ITEM_BUCKETS
    item_one_hot[_bucket(getattr(item, "nom", type(item).__name__), LEGACY_ITEM_BUCKETS)] = 1.0
    return np.asarray(base + item_features + item_one_hot, dtype=np.float32)


def extract_item_activation_observation(joueur, jeu, item, card, hook):
    base = extract_break_observation(joueur, jeu).tolist()
    item_index, card_power, card_damage, item_features = _common_item_features(joueur, item, card, hook)
    item_features.extend([
        item_index / max(1.0, float(len(joueur.objets) - 1)),
        joueur.pv_total / 30.0,
        len([obj for obj in joueur.objets if getattr(obj, "intact", False)]) / 12.0,
        1.0 if card_damage > 0 and card_damage < joueur.pv_total <= card_damage + 2 else 0.0,
    ])
    item_one_hot = [0.0] * ITEM_CLASS_BUCKETS
    class_index = _item_class_index(item)
    if 0 <= class_index < ITEM_CLASS_BUCKETS:
        item_one_hot[class_index] = 1.0
    else:
        item_one_hot[_bucket(getattr(item, "nom", type(item).__name__), ITEM_CLASS_BUCKETS)] = 1.0
    hook_one_hot = [0.0] * HOOK_BUCKETS
    hook_one_hot[HOOK_ORDER.get(hook, HOOK_BUCKETS - 1)] = 1.0
    return np.asarray(base + item_features + item_one_hot + hook_one_hot, dtype=np.float32)


def legacy_observation_size():
    return break_observation_size() + LEGACY_ITEM_EXTRA_FEATURES + LEGACY_ITEM_BUCKETS


def observation_size():
    return break_observation_size() + ITEM_EXTRA_FEATURES + ITEM_CLASS_BUCKETS + HOOK_BUCKETS
