import json
import random
from dataclasses import dataclass

from policies import HeuristicPolicy


@dataclass(frozen=True)
class LeagueEntry:
    name: str
    weight: float = 1.0


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
        self.policy_factory = policy_factory or (lambda name: HeuristicPolicy(name))
        self.rng = rng or random
        self._cache = {}

    @classmethod
    def from_json(cls, path, policy_factory=None, rng=None):
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return cls(payload.get("opponents", []), policy_factory=policy_factory, rng=rng)

    def sample(self):
        names = [entry.name for entry in self.entries]
        weights = [entry.weight for entry in self.entries]
        name = self.rng.choices(names, weights=weights, k=1)[0]
        if name not in self._cache:
            self._cache[name] = self.policy_factory(name)
        return self._cache[name]

    @staticmethod
    def _coerce_entry(entry):
        if isinstance(entry, LeagueEntry):
            return entry
        if isinstance(entry, str):
            return LeagueEntry(entry, 1.0)
        return LeagueEntry(str(entry["name"]), float(entry.get("weight", 1.0)))


class FixedPolicySampler:
    def __init__(self, policy):
        self.policy = policy

    def sample(self):
        return self.policy
