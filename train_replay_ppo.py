import argparse
import os

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from gym_replay_env import GymReplayEnv
from train_replay_model import collect_examples


def parse_net_arch(text):
    return [int(part) for part in text.split(",") if part.strip()]


def behavior_clone_actor(model, samples, seed, epochs, lr):
    if samples <= 0 or epochs <= 0:
        return
    x, y = collect_examples(samples, seed)
    obs = torch.as_tensor(x, dtype=torch.float32, device=model.device)
    labels = torch.as_tensor(y.astype(np.int64), dtype=torch.long, device=model.device)
    positives = max(1.0, float(np.sum(y)))
    negatives = max(1.0, float(len(y) - np.sum(y)))
    class_weights = torch.as_tensor(
        [1.0, negatives / positives], dtype=torch.float32, device=model.device
    )
    params = list(model.policy.mlp_extractor.policy_net.parameters())
    params.extend(model.policy.action_net.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    batch_size = 256
    rng = np.random.default_rng(seed)
    model.policy.train()
    for epoch in range(epochs):
        order = rng.permutation(len(labels))
        losses = []
        correct = 0
        for start in range(0, len(labels), batch_size):
            idx = torch.as_tensor(order[start : start + batch_size], device=model.device)
            batch_obs = obs[idx]
            batch_labels = labels[idx]
            features = model.policy.extract_features(batch_obs)
            latent_pi, _ = model.policy.mlp_extractor(features)
            logits = model.policy.action_net(latent_pi)
            loss = torch.nn.functional.cross_entropy(logits, batch_labels, weight=class_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            correct += int((torch.argmax(logits, dim=1) == batch_labels).sum().detach().cpu())
        print(
            f"bc epoch {epoch + 1}/{epochs}: "
            f"loss={np.mean(losses):.4f} acc={correct / len(labels):.3f} "
            f"positive={positives / len(y):.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Train a PPO policy for replay/pass.")
    parser.add_argument("--timesteps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=600000)
    parser.add_argument("--out", default="replay_ppo.zip")
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--bc-samples", type=int, default=0)
    parser.add_argument("--bc-epochs", type=int, default=5)
    parser.add_argument("--bc-lr", type=float, default=0.001)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--net-arch", default="128,128")
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    env = Monitor(GymReplayEnv(seed_start=args.seed))
    if args.check_env:
        check_env(GymReplayEnv(seed_start=args.seed), warn=True)

    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        verbose=1,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=0.99,
        learning_rate=3e-4,
        device=args.device,
        policy_kwargs={"net_arch": {"pi": parse_net_arch(args.net_arch), "vf": parse_net_arch(args.net_arch)}},
    )
    behavior_clone_actor(model, args.bc_samples, args.seed + 100000, args.bc_epochs, args.bc_lr)
    if args.timesteps > 0:
        model.learn(total_timesteps=args.timesteps, progress_bar=False)
    model.save(args.out)
    print(f"wrote {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
