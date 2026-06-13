import hashlib
import math

import numpy as np

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

POWER_BINS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def _stable_bucket(text, size):
    raw = hashlib.sha1(text.encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(raw[:4], "little") % size


def hero_bucket(joueur, size=24):
    return _stable_bucket(getattr(joueur.perso_obj, "nom", joueur.personnage_nom), size)


def _remaining_cards(jeu):
    donjon = jeu.donjon
    return [donjon.cartes[i] for i in donjon.ordre[donjon.index:]]


def _score_fast(joueur):
    if hasattr(joueur, "_score_rapide"):
        return joueur._score_rapide()
    return len(getattr(joueur, "pile_monstres_vaincus", []))


def extract_flee_observation(joueur, jeu):
    """Fixed-size numeric vector for the flee/continue decision."""
    cards = _remaining_cards(jeu)
    n_cards = len(cards)
    intact_objects = [o for o in joueur.objets if getattr(o, "intact", False)]
    covered_types, covered_powers = joueur._couverture_objets()

    type_counts = dict.fromkeys(MONSTER_TYPES, 0)
    power_hist = dict.fromkeys(POWER_BINS, 0)
    event_count = 0
    max_power = 0
    lethal_count = 0
    executable_count = 0
    for card in cards:
        if getattr(card, "event", False):
            event_count += 1
            continue
        power = int(getattr(card, "puissance_initiale", getattr(card, "puissance", 0)))
        max_power = max(max_power, power)
        power_hist[min(10, max(0, power))] += 1
        for t in getattr(card, "types", []):
            if t in type_counts:
                type_counts[t] += 1
        if joueur._degats_attendus(card, jeu) >= joueur.pv_total:
            lethal_count += 1
        if joueur.peut_executer_facilement(card, (covered_types, covered_powers)):
            executable_count += 1

    known = joueur.connait_prochaine_carte(jeu)
    known_is_event = 1.0 if known is not None and getattr(known, "event", False) else 0.0
    known_power = 0.0 if known is None or known_is_event else getattr(known, "puissance", 0) / 10.0
    known_lethal = 0.0
    if known is not None and not known_is_event:
        known_lethal = 1.0 if joueur._degats_attendus(known, jeu) >= joueur.pv_total else 0.0

    opponents = [j for j in jeu.joueurs if j is not joueur]
    opponent_scores = [_score_fast(j) for j in opponents if j.vivant]
    opponent_best = max(opponent_scores, default=0)
    opponent_avg = sum(opponent_scores) / len(opponent_scores) if opponent_scores else 0.0
    alive_count = sum(1 for j in jeu.joueurs if j.vivant)
    dungeon_count = sum(1 for j in jeu.joueurs if j.dans_le_dj)

    h_bucket = hero_bucket(joueur)
    hero_one_hot = [0.0] * 24
    hero_one_hot[h_bucket] = 1.0

    features = [
        joueur.pv_total / 30.0,
        _score_fast(joueur) / 30.0,
        joueur.medailles / 5.0,
        joueur.tour / 20.0,
        getattr(joueur.perso_obj, "level", 1) / 2.0,
        joueur.calculer_modificateurs() / 10.0,
        len(joueur.objets) / 10.0,
        len(intact_objects) / 10.0,
        joueur._nb_options_combat() / 10.0,
        n_cards / 60.0,
        event_count / 20.0,
        lethal_count / max(1, n_cards),
        executable_count / max(1, n_cards),
        max_power / 10.0,
        known_is_event,
        known_power,
        known_lethal,
        opponent_best / 30.0,
        opponent_avg / 30.0,
        alive_count / max(1, len(jeu.joueurs)),
        dungeon_count / max(1, len(jeu.joueurs)),
    ]
    features.extend(type_counts[t] / max(1, n_cards) for t in MONSTER_TYPES)
    features.extend(power_hist[p] / max(1, n_cards) for p in POWER_BINS)
    features.extend(hero_one_hot)
    return np.asarray(features, dtype=np.float32)


def observation_size():
    return 21 + len(MONSTER_TYPES) + len(POWER_BINS) + 24


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)
