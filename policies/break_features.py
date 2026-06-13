import numpy as np

from .replay_features import extract_replay_observation, observation_size as replay_observation_size

MAX_OBJECTS = 12
OBJECT_FEATURES = 9


def _remaining_cards(jeu):
    donjon = jeu.donjon
    return [donjon.cartes[i] for i in donjon.ordre[donjon.index:]]


def _target_count(objet, cards):
    if not (getattr(objet, "types_tags", None) or getattr(objet, "puissance_tags", None)):
        return 0
    return sum(
        1
        for card in cards
        if any(t in getattr(card, "types_initiaux", ()) for t in objet.types_tags)
        or getattr(card, "puissance_initiale", None) in objet.puissance_tags
    )


def legal_break_actions(joueur):
    return [i for i, objet in enumerate(joueur.objets[:MAX_OBJECTS]) if getattr(objet, "intact", False)]


def extract_break_observation(joueur, jeu):
    cards = _remaining_cards(jeu)
    base = extract_replay_observation(joueur, jeu).tolist()
    features = []
    for i in range(MAX_OBJECTS):
        if i >= len(joueur.objets):
            features.extend([0.0] * OBJECT_FEATURES)
            continue
        objet = joueur.objets[i]
        target_count = _target_count(objet, cards)
        features.extend(
            [
                1.0,
                1.0 if getattr(objet, "intact", False) else 0.0,
                getattr(objet, "pv_bonus", 0) / 10.0,
                getattr(objet, "modificateur_de", 0) / 10.0,
                getattr(objet, "priorite", 0) / 100.0,
                1.0 if getattr(objet, "actif", False) else 0.0,
                1.0 if getattr(objet, "non_combattant", False) else 0.0,
                target_count / max(1, len(cards)),
                1.0 if getattr(objet, "pv_bonus", 0) >= joueur.pv_total else 0.0,
            ]
        )
    return np.asarray(base + features, dtype=np.float32)


def observation_size():
    return replay_observation_size() + MAX_OBJECTS * OBJECT_FEATURES
