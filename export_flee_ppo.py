import argparse
import json

from stable_baselines3 import PPO


def tensor_to_list(state, key):
    return state[key].detach().cpu().numpy().astype(float).tolist()


def main():
    parser = argparse.ArgumentParser(description="Export a PPO flee actor to numpy JSON.")
    parser.add_argument("--model", default="flee_ppo.zip")
    parser.add_argument("--out", default="flee_ppo_policy.json")
    args = parser.parse_args()

    model = PPO.load(args.model, device="cpu")
    state = model.policy.state_dict()
    payload = {
        "type": "sb3_ppo_actor_tanh",
        "policy_layers": [
            {
                "weight": tensor_to_list(state, "mlp_extractor.policy_net.0.weight"),
                "bias": tensor_to_list(state, "mlp_extractor.policy_net.0.bias"),
            },
            {
                "weight": tensor_to_list(state, "mlp_extractor.policy_net.2.weight"),
                "bias": tensor_to_list(state, "mlp_extractor.policy_net.2.bias"),
            },
        ],
        "action_weight": tensor_to_list(state, "action_net.weight"),
        "action_bias": tensor_to_list(state, "action_net.bias"),
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
