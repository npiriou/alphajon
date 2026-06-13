import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from item_env import ItemActivationEnv
from policies import CombinedPolicy, HeuristicPolicy, NumpyBreakPolicy, NumpyPPOFleePolicy, NumpyReplayPolicy
from policies.item_features import observation_size


def build_non_item_rollout_policy(mode):
    if mode == "heuristic":
        return HeuristicPolicy("ev")
    flee = NumpyPPOFleePolicy("flee_ppo_policy.json")
    replay = NumpyReplayPolicy("replay_ppo_policy.json", flee_policy=flee)
    break_policy = NumpyBreakPolicy("break_bc_mlp_policy.json", flee_policy=flee, replay_policy=replay)
    return CombinedPolicy(flee_policy=flee, replay_policy=replay, break_policy=break_policy, item_policy=HeuristicPolicy("ev"))


class GymItemActivationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        nb_joueurs=4,
        controlled_seat=0,
        pv_min_fuite=6,
        seed_start=9000000,
        rollout_policy="current",
    ):
        super().__init__()
        self.base_env = ItemActivationEnv(
            nb_joueurs=nb_joueurs,
            controlled_seat=controlled_seat,
            pv_min_fuite=pv_min_fuite,
        )
        self.base_env.rollout_policy = build_non_item_rollout_policy(rollout_policy)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size(),),
            dtype=np.float32,
        )
        self.seed_start = int(seed_start)
        self.next_seed = int(seed_start)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.next_seed = int(seed)
        for _ in range(100):
            obs, info = self.base_env.reset(seed=self.next_seed)
            self.next_seed += 1
            if self.base_env.terminal_players is None:
                return obs.astype(np.float32), info
        fallback_seed = random.randrange(2**31)
        obs, info = self.base_env.reset(seed=fallback_seed)
        return obs.astype(np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.base_env.step(int(action))
        if terminated:
            reward = self._terminal_reward(info)
        return obs.astype(np.float32), float(reward), terminated, truncated, info

    def _terminal_reward(self, info):
        reward = 10.0 if info["win"] else -1.0
        if info["death"]:
            reward -= 3.0
        reward += 0.10 * float(info["score"])
        if info["cleared"]:
            reward += 0.5
        return reward
