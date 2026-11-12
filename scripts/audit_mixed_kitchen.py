"""Reproduce task-quality checks without training or using a GPU."""

import argparse
import json
import platform
import statistics
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from overcooked_ai_py.cooperative import KitchenConfig, MixedKitchen
from overcooked_ai_py.cooperative.controller import ReferenceController
from overcooked_ai_py.cooperative.diagnostics import overlap_probe, single_category_score_bounds


def run_episode(config, seed, controller):
    env = MixedKitchen(config)
    env.reset(seed=seed)
    for _ in range(config.horizon):
        _, _, done, _, _ = env.step(controller(env))
        if done:
            break
    m = env.state.metrics
    return {"seed": seed, "active": list(config.active_players), "score": env.state.score,
            "normal": m["normal_completed"], "joint": m["joint_completed"],
            "steps": env.state.tick, "missed": m["normal_missed"] + m["joint_missed"],
            "useful_by_agent": [m["loads_by_agent"][i] + m["serves_by_agent"][i] +
                                m["joint_ticks_by_agent"][i] for i in range(3)],
            "fingerprint": env.fingerprint}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("demo_outputs/mixed_kitchen/audit.json"))
    args = parser.parse_args()
    if args.seeds <= 0:
        parser.error("--seeds must be positive")
    config = KitchenConfig()
    results, aggregates = [], {}
    started = time.perf_counter()
    for layout in ("compact", "offset", "bottleneck"):
        aggregates[layout] = {}
        for active in ((0, 1, 2), (0, 1), (0, 2), (1, 2)):
            rows = [run_episode(replace(config, layout=layout, active_players=active), seed,
                                ReferenceController()) for seed in range(args.seeds)]
            results.extend(dict(row, layout=layout) for row in rows)
            scores = [r["score"] for r in rows]
            summary = {"mean_score": statistics.mean(scores), "min_score": min(scores),
                       "max_score": max(scores), "standard_deviation": statistics.pstdev(scores),
                       "both_types_completed_every_episode": all(r["normal"] and r["joint"] for r in rows),
                       "every_active_chef_contributed_every_episode": all(
                           all(r["useful_by_agent"][i] > 0 for i in active) for r in rows)}
            aggregates[layout]["-".join(map(str, active))] = summary
            print(layout, active, json.dumps(summary), flush=True)
    reference_elapsed = time.perf_counter() - started
    three = overlap_probe()
    two = overlap_probe((0, 1))
    assert three["score"] == 60 and two["score"] == 25
    for layout, rows in aggregates.items():
        assert rows["0-1-2"]["both_types_completed_every_episode"], layout
        assert rows["0-1-2"]["every_active_chef_contributed_every_episode"], layout
        assert rows["0-1-2"]["mean_score"] > max(
            rows[key]["mean_score"] for key in ("0-1", "0-2", "1-2")
        ), "Third-chef throughput advantage not established on " + layout
    spam = run_episode(config, 0, lambda _: (5, 5, 5))
    assert spam["normal"] == 0 and spam["joint"] == 0
    rng = np.random.default_rng(123)
    env = MixedKitchen()
    throughput_start = time.perf_counter()
    for _ in range(5000):
        _, _, done, _, _ = env.step(rng.integers(0, 6, 3))
        if done:
            env.reset()
    seconds = time.perf_counter() - throughput_start
    report = {"version": "MixedKitchen-v1", "default_config": config.to_dict(),
              "python": platform.python_version(), "numpy": np.__version__,
              "seed_count": args.seeds, "aggregates": aggregates, "episodes": results,
              "overlap_probe": {"three": three, "two": two},
              "single_category_optimistic_bounds": single_category_score_bounds(config),
              "always_interact": spam, "reference_audit_seconds": reference_elapsed,
              "environment_steps_per_second": 5000 / seconds,
              "scope": "Reference controllers and a controlled capacity proof; not optimal-policy or learned-MARL results."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Controlled overlap:", three["score"], "vs", two["score"])
    print("Environment steps/s:", round(5000 / seconds))
    print("Report:", args.output)


if __name__ == "__main__":
    main()
