import random

import numpy as np

from heros import Prophete, persos_disponibles
from joueurs import Joueur
from monstres import DonjonDeck
from objets import objets_disponibles
from policies import HeuristicPolicy
from policies.scry_features import extract_scry_observation, observation_size
from simu import ordonnanceur


class NeedScryWindowAction(Exception):
    def __init__(self, observation, info):
        super().__init__("scry window action required")
        self.observation = observation
        self.info = info


class _ScryControlledPolicy:
    def __init__(self, env):
        self.env = env
        self.index = 0

    def decide_flee(self, state, legal_actions):
        return self.env.rollout_policy.decide_flee(state, legal_actions)

    def decide_replay(self, state, legal_actions):
        return self.env.rollout_policy.decide_replay(state, legal_actions)

    def choose_item_to_break(self, state, legal_actions):
        return self.env.rollout_policy.choose_item_to_break(state, legal_actions)

    def choose_item_activation(self, state, legal_actions):
        return self.env.rollout_policy.choose_item_activation(state, legal_actions)

    def choose_scry_window_action(self, state, legal_actions):
        if self.index < len(self.env._actions):
            action = int(self.env._actions[self.index])
            self.index += 1
            return action if action in legal_actions else legal_actions[0]
        if self.env.continue_with_rollout_policy:
            return self.env.rollout_policy.choose_scry_window_action(state, legal_actions)
        obs = extract_scry_observation(state["player"], state["game"], state["cards"], state.get("source", ""))
        info = self.env._decision_info(state, legal_actions)
        info["baseline_action"] = int(self.env.rollout_policy.choose_scry_window_action(state, legal_actions))
        raise NeedScryWindowAction(obs, info)


class ScryDecisionEnv:
    action_space_n = 4
    observation_shape = (observation_size(),)

    def __init__(self, nb_joueurs=4, controlled_seat=0, pv_min_fuite=6, controlled_initial_pv=None):
        self.nb_joueurs = nb_joueurs
        self.controlled_seat = controlled_seat
        self.pv_min_fuite = pv_min_fuite
        self.controlled_initial_pv = None if controlled_initial_pv is None else int(controlled_initial_pv)
        self.seed_value = None
        self._actions = []
        self.rollout_policy = HeuristicPolicy("ev")
        self.continue_with_rollout_policy = False
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
        self._actions.append(int(action))
        obs, info = self._replay()
        if self.terminal_players is None:
            return obs, 0.0, False, False, info
        return obs, self._reward(), True, False, info

    def run_to_terminal(self, seed, actions, rollout_policy=None):
        previous_policy = self.rollout_policy
        previous_continue = self.continue_with_rollout_policy
        try:
            self.seed_value = int(seed)
            self._actions = list(actions)
            self.terminal_players = None
            self.vainqueur = None
            if rollout_policy is not None:
                self.rollout_policy = rollout_policy
            self.continue_with_rollout_policy = True
            obs, info = self._replay()
            if self.terminal_players is None:
                raise RuntimeError("rollout policy did not resolve all scry decisions")
            return obs, self._reward(), info
        finally:
            self.rollout_policy = previous_policy
            self.continue_with_rollout_policy = previous_continue

    def _build_game(self):
        random.seed(self.seed_value)
        np.random.seed(self.seed_value & 0xFFFFFFFF)
        objets_simu = list(objets_disponibles)
        for obj in objets_simu:
            obj.repare()
        noms = ["Sagarex", "Francis", "Mastho", "Mr.Adam"][: self.nb_joueurs]
        persos_pool = [p for p in persos_disponibles if type(p).__name__ != "Prophete"]
        persos = random.sample(persos_pool, self.nb_joueurs - 1)
        persos.insert(self.controlled_seat, Prophete(level=2))
        joueurs = []
        for i, nom in enumerate(noms):
            objs = random.sample(objets_simu, 6)
            for obj in objs:
                objets_simu.remove(obj)
            joueur = Joueur(nom, persos[i], objs)
            if i == self.controlled_seat and self.controlled_initial_pv is not None:
                joueur.pv_total = max(1, int(self.controlled_initial_pv))
            joueur.politique_fuite = "ev"
            joueur.policy = _ScryControlledPolicy(self) if i == self.controlled_seat else HeuristicPolicy("ev")
            joueurs.append(joueur)
        return joueurs, objets_simu

    def _replay(self):
        joueurs, objets_simu = self._build_game()
        try:
            vainqueur, joueurs_finaux = ordonnanceur(
                joueurs, DonjonDeck(), self.pv_min_fuite, objets_simu, False
            )
        except NeedScryWindowAction as exc:
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

    def _decision_info(self, state, legal_actions):
        joueur = state["player"]
        return {
            "seed": self.seed_value,
            "decision_index": len(self._actions),
            "player": joueur.nom,
            "source": state.get("source", ""),
            "pv": joueur.pv_total,
            "score": joueur._score_rapide(),
            "cards": [getattr(c, "titre", "") for c in state.get("cards", [])],
            "legal_actions": list(legal_actions),
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
        return float(reward)
