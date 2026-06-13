import argparse
import json

from stable_baselines3 import PPO

from policies.item_features import FEATURE_VERSION, observation_size


def tensor_to_list(state, key):
    return state[key].detach().cpu().numpy().astype(float).tolist()


def main():
    parser = argparse.ArgumentParser(description="Export a PPO item actor to numpy JSON.")
    parser.add_argument("--model", default="item_ppo.zip")
    parser.add_argument("--out", default="item_ppo_policy.json")
    args = parser.parse_args()

    model = PPO.load(args.model, device="cpu")
    state = model.policy.state_dict()
    policy_layers = []
    idx = 0
    while f"mlp_extractor.policy_net.{idx}.weight" in state:
        policy_layers.append(
            {
                "weight": tensor_to_list(state, f"mlp_extractor.policy_net.{idx}.weight"),
                "bias": tensor_to_list(state, f"mlp_extractor.policy_net.{idx}.bias"),
            }
        )
        idx += 2
    payload = {
        "type": "item_activation_actor_tanh",
        "feature_version": FEATURE_VERSION,
        "observation_size": observation_size(),
        "policy_layers": policy_layers,
        "action_weight": tensor_to_list(state, "action_net.weight"),
        "action_bias": tensor_to_list(state, "action_net.bias"),
        "metadata": {
            "source": args.model,
            "trainer": "train_item_ppo.py",
        },
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
