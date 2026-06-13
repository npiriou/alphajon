import numpy as np

from .flee_features import extract_flee_observation, observation_size as flee_observation_size


def _next_card_features(joueur, jeu):
    donjon = jeu.donjon
    if donjon.vide:
        return [0.0, 0.0, 0.0, 0.0]
    card = donjon.cartes[donjon.ordre[donjon.index]]
    is_event = 1.0 if getattr(card, "event", False) else 0.0
    if is_event:
        return [1.0, 0.0, 0.0, 0.0]
    power = getattr(card, "puissance_initiale", getattr(card, "puissance", 0))
    lethal = 1.0 if joueur._degats_attendus(card, jeu) >= joueur.pv_total else 0.0
    executable = 1.0 if joueur.peut_executer_facilement(card) else 0.0
    return [0.0, power / 10.0, lethal, executable]


def extract_replay_observation(joueur, jeu):
    base = extract_flee_observation(joueur, jeu).tolist()
    extras = [
        joueur.monstres_ajoutes_ce_tour / 5.0,
        1.0 if jeu.execute_next_monster else 0.0,
        1.0 if jeu.traquenard_actif else 0.0,
        1.0 if joueur.doit_passer else 0.0,
    ]
    extras.extend(_next_card_features(joueur, jeu))
    return np.asarray(base + extras, dtype=np.float32)


def observation_size():
    return flee_observation_size() + 8
