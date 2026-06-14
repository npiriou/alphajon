import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from full_decision_env import DECISION_KIND_IDS, FEATURE_SIZE, MAX_ACTIONS, FullDecisionEnv
from league_policy import LeaguePolicySampler


class GymFullDecisionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        nb_joueurs=4,
        controlled_seat=0,
        pv_min_fuite=6,
        seed_start=12000000,
        opponent_league=None,
    ):
        super().__init__()
        opponent_policy_sampler = LeaguePolicySampler.from_json(opponent_league) if opponent_league else None
        self.base_env = FullDecisionEnv(
            nb_joueurs=nb_joueurs,
            controlled_seat=controlled_seat,
            pv_min_fuite=pv_min_fuite,
            opponent_policy_sampler=opponent_policy_sampler,
        )
        self.seed_start = int(seed_start)
        self.next_seed = int(seed_start)
        self.action_space = spaces.Discrete(MAX_ACTIONS)
        self.observation_space = spaces.Dict(
            {
                "kind": spaces.Discrete(len(DECISION_KIND_IDS) + 1),
                "features": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(FEATURE_SIZE,),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(MAX_ACTIONS,),
                    dtype=np.float32,
                ),
            }
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.next_seed = int(seed)
        for _ in range(100):
            obs, info = self.base_env.reset(seed=self.next_seed)
            self.next_seed += 1
            if not info.get("terminal"):
                return self._cast_obs(obs), info
        obs, info = self.base_env.reset(seed=random.randrange(2**31))
        return self._cast_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.base_env.step(int(action))
        return self._cast_obs(obs), float(reward), bool(terminated), bool(truncated), info

    def _cast_obs(self, obs):
        return {
            "kind": int(obs["kind"]),
            "features": np.asarray(obs["features"], dtype=np.float32),
            "action_mask": np.asarray(obs["action_mask"], dtype=np.float32),
        }
