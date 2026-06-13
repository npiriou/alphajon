import argparse
import math
import multiprocessing
import os
import random

import numpy as np

from heros import persos_disponibles
from joueurs import Joueur
from monstres import DonjonDeck
from objets import objets_disponibles
from policies import (
    HeuristicPolicy,
    CombinedPolicy,
    NumpyBreakPolicy,
    NumpyItemActivationPolicy,
    ModelPolicy,
    NumpyPPOFleePolicy,
    NumpyReplayPolicy,
    RandomPolicy,
    StableBaselinesFleePolicy,
)
from simu import ordonnanceur

_WORKER_POLICY_NAMES = None
_WORKER_POLICY_CACHE = None


def make_policy(name):
    if name == "ev":
        return HeuristicPolicy("ev")
    if name == "seuils":
        return HeuristicPolicy("seuils")
    if name == "random":
        return RandomPolicy(0.5)
    if name.startswith("model:"):
        return ModelPolicy(name.split(":", 1)[1])
    if name.startswith("fastppo:"):
        return NumpyPPOFleePolicy(name.split(":", 1)[1])
    if name.startswith("replaymodel:"):
        return NumpyReplayPolicy(name.split(":", 1)[1])
    if name.startswith("breakmodel:"):
        return NumpyBreakPolicy(name.split(":", 1)[1])
    if name.startswith("itemmodel:"):
        return NumpyItemActivationPolicy(name.split(":", 1)[1])
    if name.startswith("combined:"):
        parts = name.split(":", 1)[1].split(",")
        if len(parts) not in (3, 4):
            raise ValueError("combined policy must be combined:flee_path,replay_path,break_path[,item_path]")
        flee_path, replay_path, break_path = parts[:3]
        item_policy = NumpyItemActivationPolicy(parts[3]) if len(parts) == 4 else None
        return CombinedPolicy(
            flee_policy=NumpyPPOFleePolicy(flee_path),
            replay_policy=NumpyReplayPolicy(replay_path),
            break_policy=NumpyBreakPolicy(break_path),
            item_policy=item_policy,
        )
    if name.startswith("ppo:"):
        return StableBaselinesFleePolicy(name.split(":", 1)[1])
    raise ValueError(f"unknown policy {name}")


def make_policy_cache(policy_names):
    return {name: make_policy(name) for name in policy_names}


def empty_stats(policy_names):
    return {
        p: {
            "played": 0,
            "win": 0,
            "death": 0,
            "fled": 0,
            "cleared": 0,
            "replay_decisions": 0,
            "replay_draws": 0,
            "break_decisions": 0,
            "item_activation_decisions": 0,
            "item_activations": 0,
            "item_hook_decisions": {},
            "item_hook_activations": {},
            "score": 0.0,
            "score_values": [],
        }
        for p in policy_names
    }


def _run_benchmark_with_cache(policy_names, games, seed_start, policy_cache, game_offset=0):
    stats = empty_stats(policy_names)
    for game_idx in range(games):
        seed = seed_start + game_idx
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        objets_simu = list(objets_disponibles)
        for obj in objets_simu:
            obj.repare()
        nb_joueurs = random.choice([3, 4])
        noms = ["Sagarex", "Francis", "Mastho", "Mr.Adam"][:nb_joueurs]
        persos = random.sample(persos_disponibles, nb_joueurs)
        offset = (game_offset + game_idx) % len(policy_names)
        assigned = [policy_names[(offset + i) % len(policy_names)] for i in range(nb_joueurs)]
        random.shuffle(assigned)
        joueurs = []
        for i, nom in enumerate(noms):
            objs = random.sample(objets_simu, 6)
            for obj in objs:
                objets_simu.remove(obj)
            j = Joueur(nom, persos[i], objs)
            j.policy_name = assigned[i]
            j.policy = policy_cache[assigned[i]]
            joueurs.append(j)
        vainqueur, _ = ordonnanceur(joueurs, DonjonDeck(), 6, objets_simu, False)
        for j in joueurs:
            s = stats[j.policy_name]
            s["played"] += 1
            s["win"] += int(j is vainqueur)
            s["death"] += int(not j.vivant)
            s["fled"] += int(j.fuite_reussie)
            s["cleared"] += int(j.dans_le_dj)
            s["replay_decisions"] += getattr(j, "replay_decisions", 0)
            s["replay_draws"] += getattr(j, "replay_draws", 0)
            s["break_decisions"] += getattr(j, "break_decisions", 0)
            s["item_activation_decisions"] += getattr(j, "item_activation_decisions", 0)
            s["item_activations"] += getattr(j, "item_activations", 0)
            for hook, count in getattr(j, "item_hook_decisions", {}).items():
                s["item_hook_decisions"][hook] = s["item_hook_decisions"].get(hook, 0) + count
            for hook, count in getattr(j, "item_hook_activations", {}).items():
                s["item_hook_activations"][hook] = s["item_hook_activations"].get(hook, 0) + count
            score = float(j.score_final if getattr(j, "compte_au_score", False) else 0.0)
            s["score"] += score
            s["score_values"].append(score)
    return stats


def run_benchmark(policy_names, games, seed_start):
    return _run_benchmark_with_cache(
        policy_names, games, seed_start, make_policy_cache(policy_names)
    )


def merge_stats(dest, src):
    for policy, values in src.items():
        d = dest[policy]
        for key, value in values.items():
            if key == "score_values":
                d[key].extend(value)
            elif key in ("item_hook_decisions", "item_hook_activations"):
                for hook, count in value.items():
                    d[key][hook] = d[key].get(hook, 0) + count
            else:
                d[key] += value


def _init_worker(policy_names):
    global _WORKER_POLICY_NAMES, _WORKER_POLICY_CACHE
    _WORKER_POLICY_NAMES = policy_names
    _WORKER_POLICY_CACHE = make_policy_cache(policy_names)


def _batch(args):
    games, seed_start, game_offset = args
    return _run_benchmark_with_cache(
        _WORKER_POLICY_NAMES, games, seed_start, _WORKER_POLICY_CACHE, game_offset
    )


def run_benchmark_parallel(policy_names, games, seed_start, processes):
    if processes <= 1 or games <= 1:
        return run_benchmark(policy_names, games, seed_start)

    batches = min(processes * 4, games)
    base, rest = divmod(games, batches)
    jobs = []
    offset = 0
    for i in range(batches):
        count = base + (1 if i < rest else 0)
        if count <= 0:
            continue
        jobs.append((count, seed_start + offset, offset))
        offset += count

    stats = empty_stats(policy_names)
    with multiprocessing.Pool(processes, initializer=_init_worker, initargs=(policy_names,)) as pool:
        for partial in pool.imap_unordered(_batch, jobs):
            merge_stats(stats, partial)
    return stats


def pct_ci(success, n):
    if n <= 0:
        return 0.0, 0.0
    p = success / n
    return p * 100.0, 1.96 * math.sqrt(p * (1 - p) / n) * 100.0


def print_stats(stats):
    print(
        f"{'Policy':<18} {'Played':>8} {'Win%':>12} {'Death%':>8} {'Flee%':>8} "
        f"{'Clear%':>8} {'Draw%':>8} {'Breaks':>8} {'Use%':>8} {'AvgScore':>9} {'MedScore':>9}"
    )
    for name, s in stats.items():
        n = max(1, s["played"])
        win, win_ci = pct_ci(s["win"], n)
        scores = sorted(s["score_values"])
        median = scores[len(scores) // 2] if scores else 0.0
        draw_n = max(1, s["replay_decisions"])
        use_n = max(1, s["item_activation_decisions"])
        print(
            f"{name:<18} {s['played']:>8} {win:>7.2f}+/-{win_ci:<4.2f} "
            f"{s['death']/n*100:>8.2f} {s['fled']/n*100:>8.2f} "
            f"{s['cleared']/n*100:>8.2f} {s['replay_draws']/draw_n*100:>8.2f} "
            f"{s['break_decisions']/n:>8.3f} {s['item_activations']/use_n*100:>8.2f} "
            f"{s['score']/n:>9.3f} {median:>9.3f}"
        )
        hook_decisions = s.get("item_hook_decisions", {})
        if hook_decisions:
            parts = []
            for hook in sorted(hook_decisions):
                decisions = hook_decisions[hook]
                activations = s.get("item_hook_activations", {}).get(hook, 0)
                parts.append(f"{hook}={activations}/{decisions} ({activations / max(1, decisions) * 100:.1f}%)")
            print(f"{'':<18} item hooks: " + ", ".join(parts))


def main():
    parser = argparse.ArgumentParser(description="Stage 1 flee policy benchmark.")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=200000)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["ev", "random"],
        help="ev, seuils, random, model:path.json, fastppo:path.json, replaymodel:path.json, breakmodel:path.json, combined:flee,replay,break, or ppo:path.zip",
    )
    args = parser.parse_args()
    processes = args.processes
    if processes == 0:
        processes = max(1, (os.cpu_count() or 2) - 1)
    stats = run_benchmark_parallel(args.policies, args.games, args.seed_start, processes)
    print_stats(stats)


if __name__ == "__main__":
    main()
