import random

import numpy as np

from heros import Prophete, persos_disponibles
from item_env import SCRY_TRAINING_ITEM_CLASSES
from joueurs import Joueur
from monstres import DonjonDeck
from objets import objets_disponibles
from policies import HeuristicPolicy
from policies.item_features import SCRY_HERO_CLASSES
from policies.joint_features import (
    DECISION_ITEM_ACTIVATION,
    DECISION_SCRY_WINDOW,
    extract_joint_state_observation,
    state_observation_size,
)
from simu import ordonnanceur


class NeedJointDecisionAction(Exception):
    def __init__(self, observation, info):
        super().__init__("joint decision action required")
        self.observation = observation
        self.info = info


class _JointControlledPolicy:
    def __init__(self, env):
        self.env = env
        self.index = 0

    def _scripted_or_none(self, legal_actions):
        if self.index >= len(self.env._actions):
            return None
        action = int(self.env._actions[self.index])
        self.index += 1
        if action in legal_actions:
            return action
        return legal_actions[0] if legal_actions else 0

    def decide_flee(self, state, legal_actions):
        return self.env.rollout_policy.decide_flee(state, legal_actions)

    def decide_replay(self, state, legal_actions):
        return self.env.rollout_policy.decide_replay(state, legal_actions)

    def choose_item_to_break(self, state, legal_actions):
        return self.env.rollout_policy.choose_item_to_break(state, legal_actions)

    def choose_item_activation(self, state, legal_actions):
        scripted = self._scripted_or_none(legal_actions)
        if scripted is not None:
            return scripted
        if self.env.continue_with_rollout_policy:
            return self.env.rollout_policy.choose_item_activation(state, legal_actions)
        obs = extract_joint_state_observation(state, DECISION_ITEM_ACTIVATION)
        info = self.env._decision_info(state, legal_actions, DECISION_ITEM_ACTIVATION)
        info["baseline_action"] = int(self.env.rollout_policy.choose_item_activation(state, legal_actions))
        raise NeedJointDecisionAction(obs, info)

    def choose_scry_window_action(self, state, legal_actions):
        scripted = self._scripted_or_none(legal_actions)
        if scripted is not None:
            return scripted
        if self.env.continue_with_rollout_policy:
            return self.env.rollout_policy.choose_scry_window_action(state, legal_actions)
        obs = extract_joint_state_observation(state, DECISION_SCRY_WINDOW)
        info = self.env._decision_info(state, legal_actions, DECISION_SCRY_WINDOW)
        info["baseline_action"] = int(self.env.rollout_policy.choose_scry_window_action(state, legal_actions))
        raise NeedJointDecisionAction(obs, info)


class JointDecisionEnv:
    observation_shape = (state_observation_size(),)

    def __init__(
        self,
        nb_joueurs=4,
        controlled_seat=0,
        pv_min_fuite=6,
        forced_item_class=None,
        scry_items_per_player=0,
        scry_hero_probability=0.0,
        force_controlled_prophete=False,
        controlled_initial_pv=None,
        opponent_policy_sampler=None,
    ):
        self.nb_joueurs = nb_joueurs
        self.controlled_seat = controlled_seat
        self.pv_min_fuite = pv_min_fuite
        self.forced_item_class = forced_item_class
        self.scry_items_per_player = int(scry_items_per_player)
        self.scry_hero_probability = float(scry_hero_probability)
        self.force_controlled_prophete = bool(force_controlled_prophete)
        self.controlled_initial_pv = None if controlled_initial_pv is None else int(controlled_initial_pv)
        self.opponent_policy_sampler = opponent_policy_sampler
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
                raise RuntimeError("rollout policy did not resolve all joint decisions")
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
        persos_pool = list(persos_disponibles)
        persos = []
        for seat in range(self.nb_joueurs):
            if seat == self.controlled_seat and self.force_controlled_prophete:
                perso = Prophete(level=2)
                persos_pool = [p for p in persos_pool if type(p).__name__ != "Prophete"]
                persos.append(perso)
                continue
            scry_heroes = [
                p for p in persos_pool
                if type(p).__name__ in SCRY_HERO_CLASSES and getattr(p, "level", 1) == 2
            ]
            if scry_heroes and random.random() < self.scry_hero_probability:
                perso = random.choice(scry_heroes)
            else:
                perso = random.choice(persos_pool)
            persos_pool.remove(perso)
            persos.append(perso)
        joueurs = []
        for i, nom in enumerate(noms):
            objs = []
            if i == self.controlled_seat and self.forced_item_class is not None:
                forced = next((obj for obj in objets_simu if type(obj) is self.forced_item_class), None)
                if forced is not None:
                    objs.append(forced)
                    objets_simu.remove(forced)
            for _ in range(max(0, min(self.scry_items_per_player, 6 - len(objs)))):
                candidates = [obj for obj in objets_simu if type(obj).__name__ in SCRY_TRAINING_ITEM_CLASSES]
                if not candidates:
                    break
                chosen = random.choice(candidates)
                objs.append(chosen)
                objets_simu.remove(chosen)
            sampled_objs = random.sample(objets_simu, 6 - len(objs))
            objs.extend(sampled_objs)
            for obj in sampled_objs:
                objets_simu.remove(obj)
            joueur = Joueur(nom, persos[i], objs)
            if i == self.controlled_seat and self.controlled_initial_pv is not None:
                joueur.pv_total = max(1, int(self.controlled_initial_pv))
            joueur.politique_fuite = "ev"
            joueur.policy = _JointControlledPolicy(self) if i == self.controlled_seat else self._sample_opponent_policy()
            joueurs.append(joueur)
        return joueurs, objets_simu

    def _sample_opponent_policy(self):
        if self.opponent_policy_sampler is None:
            return HeuristicPolicy("ev")
        if hasattr(self.opponent_policy_sampler, "sample"):
            return self.opponent_policy_sampler.sample()
        return self.opponent_policy_sampler()

    def _replay(self):
        joueurs, objets_simu = self._build_game()
        try:
            vainqueur, joueurs_finaux = ordonnanceur(
                joueurs, DonjonDeck(), self.pv_min_fuite, objets_simu, False
            )
        except NeedJointDecisionAction as exc:
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

    def _decision_info(self, state, legal_actions, kind):
        joueur = state["player"]
        info = {
            "seed": self.seed_value,
            "decision_index": len(self._actions),
            "kind": kind,
            "player": joueur.nom,
            "pv": joueur.pv_total,
            "score": joueur._score_rapide(),
            "legal_actions": list(legal_actions),
        }
        if kind == DECISION_ITEM_ACTIVATION:
            item = state["item"]
            card = state["card"]
            info.update(
                {
                    "item": getattr(item, "nom", type(item).__name__),
                    "item_class": type(item).__name__,
                    "card": getattr(card, "titre", ""),
                    "hook": state.get("hook", ""),
                }
            )
        elif kind == DECISION_SCRY_WINDOW:
            info.update(
                {
                    "source": state.get("source", ""),
                    "cards": [getattr(c, "titre", "") for c in state.get("cards", [])],
                }
            )
        return info

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
