import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from overcooked_ai_py.cooperative.shared import INTERACT, STAY, SharedKitchen, SharedKitchenConfig
from overcooked_ai_py.cooperative.shared.controller import ScriptedController


def save_frame(env, path):
    import pygame
    frame = env.render()
    pygame.image.save(pygame.surfarray.make_surface(frame.transpose(1, 0, 2)), str(path))


def run(output, seed):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    output.mkdir(parents=True, exist_ok=True)
    env = SharedKitchen(SharedKitchenConfig(shuffle_players=False))
    env.reset(seed)
    controller = ScriptedController(env)
    save_frame(env, output / "initial.png")
    trace = []
    while not (env.terminated or env.truncated):
        actions = controller.actions(env)
        _, reward, _, _, info = env.step(actions)
        trace.append({"tick": env.state.tick, "actions": actions, "team_reward": reward, "events": info["events"]})
    if not env.truncated or not all(info["metrics"][key] for key in ("normal_completed", "joint_completed")):
        raise RuntimeError("The mixed-order example did not reach the time limit with both types of deliveries")
    save_frame(env, output / "completed.png")
    summary = {"controller": "scripted", "seed": seed, "config": asdict(env.config), "fingerprint": env.fingerprint(), "metrics": info["metrics"], "orders": info["orders"]}
    (output / "mixed_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "mixed_trajectory.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    joint = SharedKitchen(SharedKitchenConfig(normal_orders=0, joint_orders=1, shuffle_players=False))
    joint.reset(seed)
    controller = ScriptedController(joint)
    witness = None
    while not (joint.terminated or joint.truncated):
        actions = controller.actions(joint)
        cutting = joint.state.cutting
        pair = (controller.server, controller.main)
        if witness is None and cutting.item is not None and cutting.progress == 0:
            ready = joint.state.cooker.item is not None and joint.state.cooker.remaining == 0
            together = ready and all(actions[i] == INTERACT for i in pair)
            actions[controller.side] = INTERACT if together else STAY
            if together:
                save_frame(joint, output / "joint_before.png")
        _, _, _, _, detail = joint.step(actions)
        if any(event["type"] == "unload" for event in detail["events"]):
            witness = {"tick": joint.state.tick, "actions": actions, "events": detail["events"], "side_unit": joint.state.cutting.item.units[0], "side_chef": controller.side}
            save_frame(joint, output / "joint_and_chop.png")
    order = joint.order(0)
    if not joint.terminated or witness is None or order.delivered_units[1] != witness["side_unit"] or order.side_prepared_by != (controller.side,):
        raise RuntimeError("The same-meal concurrency example did not finish")
    witness["completed_order"] = asdict(order)
    witness["score"] = joint.state.score
    (output / "same_meal_evidence.json").write_text(json.dumps(witness, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "mixed": summary["metrics"], "same_meal_joint_tick": witness["tick"], "side_chef": controller.side + 1}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "demo_outputs" / "shared_kitchen")
    args = parser.parse_args()
    run(args.output, args.seed)
