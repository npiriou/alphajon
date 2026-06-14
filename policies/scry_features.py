import hashlib

import numpy as np

CARD_SLOTS = 2
MONSTER_TYPES = [
    "Gobelin",
    "Squelette",
    "Orc",
    "Vampire",
    "Golem",
    "Liche",
    "Demon",
    "Dragon",
    "Rat",
]
CARD_FEATURES = 8 + len(MONSTER_TYPES)
ITEM_CLASS_BUCKETS = 320
SOURCE_BUCKETS = 8


def _bucket(text, buckets):
    raw = hashlib.sha1(str(text).encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(raw[:4], "little") % buckets


def _score_fast(joueur):
    return joueur._score_rapide() if hasattr(joueur, "_score_rapide") else len(joueur.pile_monstres_vaincus)


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


def _add_item_feature(features, item, value):
    class_index = _item_class_index(item)
    if 0 <= class_index < ITEM_CLASS_BUCKETS:
        features[class_index] += float(value)
    else:
        features[_bucket(getattr(item, "nom", type(item).__name__), ITEM_CLASS_BUCKETS)] += float(value)


def extract_scry_observation(joueur, jeu, cards, source):
    opponents = [j for j in jeu.joueurs if j is not joueur]
    opponent_best = max((j._score_rapide() for j in opponents if j.vivant), default=0)
    own_intact = [0.0] * ITEM_CLASS_BUCKETS
    own_broken = [0.0] * ITEM_CLASS_BUCKETS
    opponent_intact = [0.0] * ITEM_CLASS_BUCKETS
    for obj in getattr(joueur, "objets", []):
        if getattr(obj, "intact", False):
            _add_item_feature(own_intact, obj, 1.0 / 6.0)
        else:
            _add_item_feature(own_broken, obj, 1.0 / 6.0)
    opponent_slots = max(1.0, float(sum(len(getattr(j, "objets", [])) for j in opponents)))
    for opponent in opponents:
        for obj in getattr(opponent, "objets", []):
            if getattr(obj, "intact", False):
                _add_item_feature(opponent_intact, obj, 1.0 / opponent_slots)
    features = [
        joueur.pv_total / 30.0,
        _score_fast(joueur) / 30.0,
        opponent_best / 30.0,
        (_score_fast(joueur) - opponent_best) / 30.0,
        joueur.medailles / 5.0,
        joueur.tour / 20.0,
        len([o for o in joueur.objets if getattr(o, "intact", False)]) / 10.0,
        (jeu.donjon.nb_cartes - jeu.donjon.index) / 60.0,
        1.0 if jeu.traquenard_actif else 0.0,
        1.0 if jeu.execute_next_monster else 0.0,
    ]
    for card in list(cards)[:CARD_SLOTS]:
        is_event = 1.0 if getattr(card, "event", False) else 0.0
        power = 0.0 if is_event else getattr(card, "puissance_initiale", getattr(card, "puissance", 0)) / 10.0
        damage = 0.0 if is_event else joueur._degats_attendus(card, jeu) / 10.0
        lethal = 1.0 if (not is_event and joueur._degats_attendus(card, jeu) >= joueur.pv_total) else 0.0
        easy = 1.0 if (not is_event and joueur.peut_executer_facilement(card)) else 0.0
        x_card = 1.0 if getattr(card, "is_X", False) else 0.0
        gold = 1.0 if getattr(card, "effet", "") == "GOLD" else 0.0
        score_value = 0.0 if is_event else (2.0 if gold else 1.0) / 2.0
        type_features = [1.0 if t in getattr(card, "types", ()) else 0.0 for t in MONSTER_TYPES]
        features.extend([1.0, is_event, power, damage, lethal, easy, x_card, score_value] + type_features)
    missing = CARD_SLOTS - min(CARD_SLOTS, len(cards))
    if missing > 0:
        features.extend([0.0] * missing * CARD_FEATURES)
    source_one_hot = [0.0] * SOURCE_BUCKETS
    source_one_hot[_bucket(source, SOURCE_BUCKETS)] = 1.0
    return np.asarray(features + own_intact + own_broken + opponent_intact + source_one_hot, dtype=np.float32)


def observation_size():
    return 10 + CARD_SLOTS * CARD_FEATURES + ITEM_CLASS_BUCKETS * 3 + SOURCE_BUCKETS
