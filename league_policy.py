import json
import random
from dataclasses import dataclass
from typing import Optional

from policies import (
    CombinedPolicy,
    HeuristicPolicy,
    HybridScryItemActivationPolicy,
    NumpyBreakPolicy,
    NumpyItemActivationPolicy,
    NumpyJointDecisionPolicy,
    NumpyPPOFleePolicy,
    NumpyReplayPolicy,
    NumpyScryWindowPolicy,
    RandomPolicy,
)


@dataclass(frozen=True)
class LeagueEntry:
    name: str
    weight: float = 1.0
    policy: Optional[str] = None


class LeaguePolicySampler:
    """Weighted sampler for opponent policies.

    The sampler is intentionally independent from the benchmark policy parser:
    callers can inject the policy factory that is appropriate for their entry
    point while envs only need the small sample() interface.
    """

    def __init__(self, entries, policy_factory=None, rng=None):
        self.entries = [self._coerce_entry(entry) for entry in entries]
        if not self.entries:
            raise ValueError("league must contain at least one opponent")
        if any(entry.weight <= 0 for entry in self.entries):
            raise ValueError("league opponent weights must be positive")
        self.policy_factory = policy_factory or make_policy_from_spec
        self.rng = rng or random
        self._cache = {}

    @classmethod
    def from_json(cls, path, policy_factory=None, rng=None):
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return cls(payload.get("opponents", []), policy_factory=policy_factory, rng=rng)

    def sample(self):
        weights = [entry.weight for entry in self.entries]
        entry = self.rng.choices(self.entries, weights=weights, k=1)[0]
        spec = entry.policy or entry.name
        if spec not in self._cache:
            self._cache[spec] = self.policy_factory(spec)
        return self._cache[spec]

    @staticmethod
    def _coerce_entry(entry):
        if isinstance(entry, LeagueEntry):
            return entry
        if isinstance(entry, str):
            return LeagueEntry(entry, 1.0)
        return LeagueEntry(
            str(entry["name"]),
            float(entry.get("weight", 1.0)),
            None if entry.get("policy") is None else str(entry["policy"]),
        )


class FixedPolicySampler:
    def __init__(self, policy):
        self.policy = policy

    def sample(self):
        return self.policy


def make_policy_from_spec(spec):
    if spec == "ev":
        return HeuristicPolicy("ev")
    if spec == "seuils":
        return HeuristicPolicy("seuils")
    if spec == "random":
        return RandomPolicy(0.5)
    if spec == "current":
        return make_policy_from_spec(
            "combined:flee_ppo_policy.json,replay_ppo_policy.json,break_bc_mlp_policy.json,item_bc_mlp_policy.json"
        )
    if spec.startswith("fastppo:"):
        return NumpyPPOFleePolicy(spec.split(":", 1)[1])
    if spec.startswith("replaymodel:"):
        return NumpyReplayPolicy(spec.split(":", 1)[1])
    if spec.startswith("breakmodel:"):
        return NumpyBreakPolicy(spec.split(":", 1)[1])
    if spec.startswith("itemmodel:"):
        return NumpyItemActivationPolicy(spec.split(":", 1)[1])
    if spec.startswith("hybriditem:"):
        base_path, scry_path = spec.split(":", 1)[1].split("+", 1)
        return HybridScryItemActivationPolicy(base_path, scry_path)
    if spec.startswith("scrywindow:"):
        return NumpyScryWindowPolicy(spec.split(":", 1)[1])
    if spec.startswith("jointq:"):
        return NumpyJointDecisionPolicy(spec.split(":", 1)[1])
    if spec.startswith("combined:"):
        parts = spec.split(":", 1)[1].split(",")
        if len(parts) not in (3, 4, 5):
            raise ValueError("combined policy must be combined:flee_path,replay_path,break_path[,item_path[,scry_path]]")
        flee_path, replay_path, break_path = parts[:3]
        flee = NumpyPPOFleePolicy(flee_path)
        replay = NumpyReplayPolicy(replay_path, flee_policy=flee)
        break_policy = NumpyBreakPolicy(break_path, flee_policy=flee, replay_policy=replay)
        item_policy = None
        if len(parts) >= 4:
            item_spec = parts[3]
            if item_spec.startswith("hybriditem:"):
                base_path, scry_path = item_spec.split(":", 1)[1].split("+", 1)
                item_policy = HybridScryItemActivationPolicy(base_path, scry_path)
            elif item_spec.startswith("jointq:"):
                item_policy = NumpyJointDecisionPolicy(item_spec.split(":", 1)[1])
            else:
                item_policy = NumpyItemActivationPolicy(
                    item_spec,
                    flee_policy=flee,
                    replay_policy=replay,
                    break_policy=break_policy,
                )
        scry_policy = None
        if len(parts) == 5:
            scry_spec = parts[4]
            if scry_spec.startswith("jointq:"):
                scry_policy = NumpyJointDecisionPolicy(scry_spec.split(":", 1)[1])
            else:
                scry_policy = NumpyScryWindowPolicy(scry_spec)
        return CombinedPolicy(
            flee_policy=flee,
            replay_policy=replay,
            break_policy=break_policy,
            item_policy=item_policy,
            scry_policy=scry_policy,
        )
    raise ValueError(f"unknown league policy spec {spec}")
