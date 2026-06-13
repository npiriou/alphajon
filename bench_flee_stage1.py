import argparse
import math
import random

import numpy as np

from heros import persos_disponibles
from joueurs import Joueur
from monstres import DonjonDeck
from objets import objets_disponibles
from policies import HeuristicPolicy, ModelPolicy, RandomPolicy, StableBaselinesFleePolicy
from simu import ordonnanceur


def make_policy(name):
    if name == "ev":
        return HeuristicPolicy("ev")
    if name == "seuils":
        return HeuristicPolicy("seuils")
    if name == "random":
        return RandomPolicy(0.5)
    if name.startswith("model:"):
        return ModelPolicy(name.split(":", 1)[1])
    if name.startswith("ppo:"):
        return StableBaselinesFleePolicy(name.split(":", 1)[1])
    raise ValueError(f"unknown policy {name}")


def empty_stats(policy_names):
    return {
        p: {
            "played": 0,
            "win": 0,
            "death": 0,
            "fled": 0,
            "cleared": 0,
            "score": 0.0,
            "score_values": [],
        }
        for p in policy_names
    }


def run_benchmark(policy_names, games, seed_start):
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
        offset = game_idx % len(policy_names)
        assigned = [policy_names[(offset + i) % len(policy_names)] for i in range(nb_joueurs)]
        random.shuffle(assigned)
        joueurs = []
        for i, nom in enumerate(noms):
            objs = random.sample(objets_simu, 6)
            for obj in objs:
                objets_simu.remove(obj)
            j = Joueur(nom, persos[i], objs)
            j.policy_name = assigned[i]
            j.policy = make_policy(assigned[i])
            joueurs.append(j)
        vainqueur, _ = ordonnanceur(joueurs, DonjonDeck(), 6, objets_simu, False)
        for j in joueurs:
            s = stats[j.policy_name]
            s["played"] += 1
            s["win"] += int(j is vainqueur)
            s["death"] += int(not j.vivant)
            s["fled"] += int(j.fuite_reussie)
            s["cleared"] += int(j.dans_le_dj)
            score = float(j.score_final if getattr(j, "compte_au_score", False) else 0.0)
            s["score"] += score
            s["score_values"].append(score)
    return stats


def pct_ci(success, n):
    if n <= 0:
        return 0.0, 0.0
    p = success / n
    return p * 100.0, 1.96 * math.sqrt(p * (1 - p) / n) * 100.0


def print_stats(stats):
    print(
        f"{'Policy':<18} {'Played':>8} {'Win%':>12} {'Death%':>8} {'Flee%':>8} "
        f"{'Clear%':>8} {'AvgScore':>9} {'MedScore':>9}"
    )
    for name, s in stats.items():
        n = max(1, s["played"])
        win, win_ci = pct_ci(s["win"], n)
        scores = sorted(s["score_values"])
        median = scores[len(scores) // 2] if scores else 0.0
        print(
            f"{name:<18} {s['played']:>8} {win:>7.2f}+/-{win_ci:<4.2f} "
            f"{s['death']/n*100:>8.2f} {s['fled']/n*100:>8.2f} "
            f"{s['cleared']/n*100:>8.2f} {s['score']/n:>9.3f} {median:>9.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Stage 1 flee policy benchmark.")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=200000)
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["ev", "random"],
        help="ev, seuils, random, or model:path/to/flee_model.json",
    )
    args = parser.parse_args()
    stats = run_benchmark(args.policies, args.games, args.seed_start)
    print_stats(stats)


if __name__ == "__main__":
    main()
