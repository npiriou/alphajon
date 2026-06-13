import random

import numpy as np

from heros import persos_disponibles
from joueurs import Joueur
from monstres import DonjonDeck
from objets import objets_disponibles
from policies import FLEE_ACTION_ATTEMPT, FLEE_ACTION_CONTINUE, HeuristicPolicy
from policies.flee_features import extract_flee_observation, observation_size
from simu import ordonnanceur


class NeedFleeAction(Exception):
    def __init__(self, observation, info):
        super().__init__("flee action required")
        self.observation = observation
        self.info = info


class _ReplayControlledPolicy:
    def __init__(self, env):
        self.env = env
        self.index = 0

    def decide_flee(self, state, legal_actions):
        if self.index < len(self.env._actions):
            action = int(self.env._actions[self.index])
            self.index += 1
            if action in legal_actions:
                return action
            return FLEE_ACTION_CONTINUE
        obs = extract_flee_observation(state["player"], state["game"])
        info = self.env._decision_info(state)
        raise NeedFleeAction(obs, info)


class FleeEnv:
    """Gymnasium-like environment for the flee/continue action.

    The current simulator is not resumable, so the env replays the episode from
    the same seed plus the action prefix until the next flee decision or terminal
    state. This keeps Stage 1 isolated from the rules engine.
    """

    action_space_n = 2
    observation_shape = (observation_size(),)

    def __init__(self, nb_joueurs=4, controlled_seat=0, pv_min_fuite=6):
        self.nb_joueurs = nb_joueurs
        self.controlled_seat = controlled_seat
        self.pv_min_fuite = pv_min_fuite
        self.seed_value = None
        self._actions = []
        self.last_obs = np.zeros(self.observation_shape, dtype=np.float32)
        self.last_info = {}
        self.terminal_players = None
        self.vainqueur = None

    def reset(self, seed=None):
        self.seed_value = random.randrange(2**31) if seed is None else int(seed)
        self._actions = []
        self.terminal_players = None
        self.vainqueur = None
        return self._replay()

    def step(self, action):
        if int(action) not in (FLEE_ACTION_CONTINUE, FLEE_ACTION_ATTEMPT):
            raise ValueError("flee action must be 0 (continue) or 1 (attempt flee)")
        self._actions.append(int(action))
        obs, info = self._replay()
        if self.terminal_players is None:
            return obs, 0.0, False, False, info
        reward = self._reward()
        return obs, reward, True, False, info

    def _build_game(self):
        random.seed(self.seed_value)
        np.random.seed(self.seed_value & 0xFFFFFFFF)
        objets_simu = list(objets_disponibles)
        for obj in objets_simu:
            obj.repare()
        noms = ["Sagarex", "Francis", "Mastho", "Mr.Adam"][: self.nb_joueurs]
        persos = random.sample(persos_disponibles, self.nb_joueurs)
        joueurs = []
        for i, nom in enumerate(noms):
            objs = random.sample(objets_simu, 6)
            for obj in objs:
                objets_simu.remove(obj)
            joueur = Joueur(nom, persos[i], objs)
            joueur.politique_fuite = "ev"
            joueur.policy = _ReplayControlledPolicy(self) if i == self.controlled_seat else HeuristicPolicy("ev")
            joueurs.append(joueur)
        return joueurs, objets_simu

    def _replay(self):
        joueurs, objets_simu = self._build_game()
        try:
            vainqueur, joueurs_finaux = ordonnanceur(
                joueurs, DonjonDeck(), self.pv_min_fuite, objets_simu, False
            )
        except NeedFleeAction as exc:
            self.last_obs = exc.observation
            self.last_info = exc.info
            self.terminal_players = None
            self.vainqueur = None
            return self.last_obs, self.last_info
        self.vainqueur = vainqueur
        self.terminal_players = joueurs_finaux
        self.last_info = self._terminal_info()
        self.last_obs = np.zeros(self.observation_shape, dtype=np.float32)
        return self.last_obs, self.last_info

    def _decision_info(self, state):
        joueur = state["player"]
        jeu = state["game"]
        return {
            "seed": self.seed_value,
            "decision_index": len(self._actions),
            "player": joueur.nom,
            "tour": joueur.tour,
            "pv": joueur.pv_total,
            "score": joueur._score_rapide(),
            "cards_remaining": jeu.donjon.nb_cartes - jeu.donjon.index,
            "legal_actions": [FLEE_ACTION_CONTINUE, FLEE_ACTION_ATTEMPT],
        }

    def _terminal_info(self):
        joueur = self.terminal_players[self.controlled_seat]
        return {
            "seed": self.seed_value,
            "terminal": True,
            "win": joueur is self.vainqueur,
            "death": not joueur.vivant,
            "fled": joueur.fuite_reussie,
            "cleared": joueur.dans_le_dj,
            "score": joueur.score_final,
            "actions": len(self._actions),
        }

    def _reward(self):
        joueur = self.terminal_players[self.controlled_seat]
        reward = 1.0 if joueur is self.vainqueur else 0.0
        if not joueur.vivant:
            reward -= 1.0
        else:
            reward += 0.1
        reward += 0.02 * joueur.score_final
        if joueur.fuite_reussie and joueur.score_final <= 2:
            reward -= 0.05
        return float(reward)
