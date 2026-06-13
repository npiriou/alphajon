import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from gym_flee_env import GymFleeEnv


def main():
    parser = argparse.ArgumentParser(description="Train a PPO policy for flee/continue.")
    parser.add_argument("--timesteps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=300000)
    parser.add_argument("--out", default="flee_ppo.zip")
    parser.add_argument("--check-env", action="store_true")
    args = parser.parse_args()

    env = Monitor(GymFleeEnv(seed_start=args.seed))
    if args.check_env:
        check_env(GymFleeEnv(seed_start=args.seed), warn=True)

    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        verbose=1,
        n_steps=256,
        batch_size=64,
        gamma=0.99,
        learning_rate=3e-4,
        device="cpu",
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    model.save(args.out)
    print(f"wrote {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
