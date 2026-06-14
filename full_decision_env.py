import random

import numpy as np

from heros import persos_disponibles
from joueurs import Joueur
from monstres import DonjonDeck
from objets import objets_disponibles
from policies import HeuristicPolicy
from policies.break_features import extract_break_observation, observation_size as break_observation_size
from policies.flee_features import extract_flee_observation, observation_size as flee_observation_size
from policies.item_features import extract_item_activation_observation, observation_size as item_observation_size
from policies.replay_features import extract_replay_observation, observation_size as replay_observation_size
from policies.scry_features import extract_scry_observation, observation_size as scry_observation_size
from simu import GameEngine

DECISION_KIND_IDS = {
    "flee": 0,
    "replay": 1,
    "break": 2,
    "item": 3,
    "scry": 4,
}
TERMINAL_KIND_ID = len(DECISION_KIND_IDS)
MAX_ACTIONS = 16
FEATURE_SIZE = max(
    flee_observation_size(),
    replay_observation_size(),
    break_observation_size(),
    item_observation_size(),
    scry_observation_size(),
)


class FullDecisionEnv:
    """Live, pausible full-decision environment backed by GameEngine.

    This env does not replay from seed/action-prefix. It keeps one GameEngine
    alive, pauses at DecisionPoint objects, applies an action, and resumes the
    same in-memory game.
    """

    observation_shape = (FEATURE_SIZE,)
    action_space_n = MAX_ACTIONS

    def __init__(self, nb_joueurs=4, controlled_seat=0, pv_min_fuite=6, opponent_policy_sampler=None):
        self.nb_joueurs = int(nb_joueurs)
        self.controlled_seat = int(controlled_seat)
        self.pv_min_fuite = int(pv_min_fuite)
        self.opponent_policy_sampler = opponent_policy_sampler
        self.seed_value = None
        self.engine = None
        self.current_decision = None
        self.last_obs = self._empty_observation()
        self.last_info = {}

    def reset(self, seed=None):
        self.seed_value = random.randrange(2**31) if seed is None else int(seed)
        joueurs, objets_simu = self._build_game()
        self.engine = GameEngine(joueurs, DonjonDeck(), self.pv_min_fuite, objets_simu)
        return self._advance_to_controlled_decision()

    def step(self, action):
        if self.engine is None or self.current_decision is None:
            raise RuntimeError("step called without a pending decision")
        requested_action = int(action)
        invalid_action = requested_action not in self.current_decision.legal_actions
        if invalid_action:
            action = self.current_decision.legal_actions[0] if self.current_decision.legal_actions else 0
        self.engine.apply_decision(int(action))
        obs, info = self._advance_to_controlled_decision()
        if invalid_action:
            info["invalid_action"] = requested_action
        if self.engine.done:
            reward = self._reward()
            if invalid_action:
                reward -= 0.05
            return obs, reward, True, False, info
        return obs, -0.05 if invalid_action else 0.0, False, False, info

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
            if i != self.controlled_seat:
                joueur.policy = self._sample_opponent_policy()
            joueurs.append(joueur)
        return joueurs, objets_simu

    def _sample_opponent_policy(self):
        if self.opponent_policy_sampler is None:
            return HeuristicPolicy("ev")
        if hasattr(self.opponent_policy_sampler, "sample"):
            return self.opponent_policy_sampler.sample()
        return self.opponent_policy_sampler()

    def _advance_to_controlled_decision(self):
        while True:
            decision = self.engine.step_until_decision(log=False)
            if decision is None:
                self.current_decision = None
                self.last_obs = self._empty_observation()
                self.last_info = self._terminal_info()
                return self.last_obs, self.last_info
            if decision.player_index == self.controlled_seat:
                self.current_decision = decision
                self.last_obs = self._observation(decision)
                self.last_info = self._decision_info(decision)
                return self.last_obs, self.last_info
            self.engine.apply_decision(self._delegate_action(decision))

    def _delegate_action(self, decision):
        policy = getattr(decision.player, "policy", None)
        if policy is not None and decision.policy_method:
            action = getattr(policy, decision.policy_method)(decision.state, decision.legal_actions)
            if int(action) in decision.legal_actions:
                return int(action)
        if decision.fallback is not None:
            action = int(decision.fallback())
            if action in decision.legal_actions:
                return action
        return int(decision.legal_actions[0]) if decision.legal_actions else 0

    def _observation(self, decision):
        features = self._features(decision)
        padded = np.zeros((FEATURE_SIZE,), dtype=np.float32)
        padded[: min(len(features), FEATURE_SIZE)] = features[:FEATURE_SIZE]
        mask = np.zeros((MAX_ACTIONS,), dtype=np.float32)
        for action in decision.legal_actions:
            action = int(action)
            if 0 <= action < MAX_ACTIONS:
                mask[action] = 1.0
        return {
            "kind": np.asarray(DECISION_KIND_IDS.get(decision.kind, TERMINAL_KIND_ID), dtype=np.int64),
            "features": padded,
            "action_mask": mask,
        }

    def _features(self, decision):
        state = decision.state
        player = decision.player
        game = decision.game
        if decision.kind == "flee":
            return extract_flee_observation(player, game)
        if decision.kind == "replay":
            return extract_replay_observation(player, game)
        if decision.kind == "break":
            return extract_break_observation(player, game)
        if decision.kind == "item":
            return extract_item_activation_observation(
                player,
                game,
                state.get("item"),
                state.get("card"),
                state.get("hook", ""),
            )
        if decision.kind == "scry":
            return extract_scry_observation(player, game, state.get("cards", []), state.get("source", ""))
        return np.zeros((FEATURE_SIZE,), dtype=np.float32)

    def _decision_info(self, decision):
        player = decision.player
        return {
            "seed": self.seed_value,
            "kind": decision.kind,
            "player_index": decision.player_index,
            "player": player.nom,
            "tour": player.tour,
            "pv": player.pv_total,
            "score": player._score_rapide(),
            "legal_actions": list(decision.legal_actions),
            "phase": decision.game.phase,
        }

    def _terminal_info(self):
        joueur = self.engine.terminal_players[self.controlled_seat]
        return {
            "seed": self.seed_value,
            "terminal": True,
            "win": joueur is self.engine.vainqueur,
            "death": not joueur.vivant,
            "fled": joueur.fuite_reussie,
            "cleared": joueur.dans_le_dj,
            "score": joueur.score_final,
        }

    def _reward(self):
        joueur = self.engine.terminal_players[self.controlled_seat]
        reward = 2.0 if joueur is self.engine.vainqueur else -0.5
        if not joueur.vivant:
            reward -= 1.0
        else:
            reward += 0.05
        reward += 0.05 * joueur.score_final
        if joueur.dans_le_dj:
            reward += 0.2
        return float(reward)

    def _empty_observation(self):
        return {
            "kind": np.asarray(TERMINAL_KIND_ID, dtype=np.int64),
            "features": np.zeros((FEATURE_SIZE,), dtype=np.float32),
            "action_mask": np.zeros((MAX_ACTIONS,), dtype=np.float32),
        }
