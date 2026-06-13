import random

import numpy as np

from heros import persos_disponibles
from joueurs import Joueur
from monstres import DonjonDeck
from objets import objets_disponibles
from policies import FLEE_ACTION_CONTINUE, HeuristicPolicy, REPLAY_ACTION_DRAW, REPLAY_ACTION_PASS
from policies.replay_features import extract_replay_observation, observation_size
from simu import ordonnanceur


class NeedReplayAction(Exception):
    def __init__(self, observation, info):
        super().__init__("replay action required")
        self.observation = observation
        self.info = info


class _ReplayControlledPolicy:
    def __init__(self, env):
        self.env = env
        self.index = 0
        self.flee_policy = HeuristicPolicy("ev")
        self.fallback = HeuristicPolicy("ev")

    def decide_flee(self, state, legal_actions):
        return self.flee_policy.decide_flee(state, legal_actions)

    def choose_item_to_break(self, state, legal_actions):
        return self.fallback.choose_item_to_break(state, legal_actions)

    def choose_item_activation(self, state, legal_actions):
        return self.fallback.choose_item_activation(state, legal_actions)

    def decide_replay(self, state, legal_actions):
        if self.index < len(self.env._actions):
            action = int(self.env._actions[self.index])
            self.index += 1
            if action in legal_actions:
                return action
            return REPLAY_ACTION_PASS
        obs = extract_replay_observation(state["player"], state["game"])
        info = self.env._decision_info(state)
        raise NeedReplayAction(obs, info)


class ReplayEnv:
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
        if int(action) not in (REPLAY_ACTION_PASS, REPLAY_ACTION_DRAW):
            raise ValueError("replay action must be 0 (pass) or 1 (draw)")
        self._actions.append(int(action))
        obs, info = self._replay()
        if self.terminal_players is None:
            return obs, 0.0, False, False, info
        return obs, self._reward(), True, False, info

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
        except NeedReplayAction as exc:
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
            "legal_actions": [REPLAY_ACTION_PASS, REPLAY_ACTION_DRAW],
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
        reward = 2.0 if joueur is self.vainqueur else -0.5
        if not joueur.vivant:
            reward -= 1.0
        else:
            reward += 0.05
        reward += 0.05 * joueur.score_final
        if joueur.dans_le_dj:
            reward += 0.2
        return float(reward)
