import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from flee_env import FleeEnv
from policies.flee_features import observation_size


class GymFleeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, nb_joueurs=4, controlled_seat=0, pv_min_fuite=6, seed_start=300000):
        super().__init__()
        self.base_env = FleeEnv(
            nb_joueurs=nb_joueurs,
            controlled_seat=controlled_seat,
            pv_min_fuite=pv_min_fuite,
        )
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
        return obs.astype(np.float32), float(reward), terminated, truncated, info
