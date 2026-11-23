"""Run paired scripted episodes and optionally export six complete delivery traces."""
import argparse
import json
from pathlib import Path

import numpy as np

from overcooked_ai_py.cooperative.partition import PartitionConfig, PartitionKitchen
from overcooked_ai_py.cooperative.partition.controller import ScriptedChefs
from overcooked_ai_py.cooperative.partition.model import RECIPE_NAMES


def evaluate(episodes, seed, traces=None, policy_factory=ScriptedChefs):
    results = {}
    for layout in ("sync", "counter"):
        delivered, expired, ages, issued = np.zeros((4, 3), dtype=int)
        totals = dict(sync_attempts=0, sync_successes=0, wait_steps=0, pickups=0)
        saved = set()
        env = PartitionKitchen(PartitionConfig(layout=layout))
        for episode in range(episodes):
            episode_seed = seed + episode
            env.reset(episode_seed)
            chefs = policy_factory(env)
            issued += np.bincount(env.state["recipe"], minlength=3)
            actions_log = []
            for _ in range(env.config.horizon):
                actions = chefs.actions()
                _, _, _, _, info = env.step(actions)
                if traces is not None and len(saved) < 3:
                    actions_log.append(actions)
                for k in totals:
                    totals[k] += int(info[k])
                for slot, recipe in enumerate(info["recipe"]):
                    if info["delivered"][slot]:
                        delivered[recipe] += 1
                        ages[recipe] += info["age"][slot]
                        if traces is not None and recipe not in saved:
                            traces.mkdir(parents=True, exist_ok=True)
                            trace = dict(layout=layout, seed=episode_seed, recipe=RECIPE_NAMES[recipe],
                                         order_id=int(info["order_ids"][slot]), actions=actions_log)
                            (traces / f"{layout}-{RECIPE_NAMES[recipe]}.json").write_text(
                                json.dumps(trace), encoding="utf-8")
                            saved.add(recipe)
                    expired[recipe] += info["expired"][slot]
                replacements = int((info["delivered"] | info["expired"]).sum())
                if replacements:
                    issued += np.bincount(env.state["recipe"][-replacements:], minlength=3)
        rows = {}
        for i, name in enumerate(RECIPE_NAMES):
            rows[name] = dict(issued=int(issued[i]), delivered=int(delivered[i]), expired=int(expired[i]),
                              pending=int(issued[i] - delivered[i] - expired[i]),
                              delivery_rate=round(delivered[i] / issued[i], 4) if issued[i] else None,
                              timeout_rate=round(expired[i] / issued[i], 4) if issued[i] else None,
                              mean_delivery_steps=round(ages[i] / delivered[i], 2) if delivered[i] else None)
        results[layout] = dict(recipes=rows, **totals,
                               mean_counter_wait=round(totals["wait_steps"] / totals["pickups"], 2)
                               if totals["pickups"] else None)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--traces", type=Path)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    print(json.dumps(evaluate(args.episodes, args.seed, args.traces), indent=2))
