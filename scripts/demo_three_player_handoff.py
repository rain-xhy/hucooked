"""Scripted end-to-end check of the ordinary and synchronous three-player maps.

This is a deterministic controller for checking rules, not a trained RL policy.
Run from the repository root: python scripts/demo_three_player_handoff.py
"""

import argparse
import json
import os
import shutil
import tempfile
from collections import deque
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import cv2
import pygame

from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=["three_player_handoff", "three_player_sync_handoff"],
                        default="three_player_sync_handoff")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir or root / "demo_outputs" / args.layout
    output.mkdir(parents=True, exist_ok=True)
    sync = args.layout == "three_player_sync_handoff"
    mdp = OvercookedGridworld.from_layout_name(args.layout)
    env = OvercookedEnv.from_mdp(mdp, horizon=400, info_level=0)
    observations = env.lossless_state_encoding_mdp(env.state)
    assert len(observations) == 3
    visualizer = StateVisualizer(tile_size=64, hud_font_size=14, hud_order_size=20)
    score = 0
    attempts = 0
    successes = 0
    wait_agent_steps = 0
    trajectory = []
    I, W = Action.INTERACT, Action.STAY

    def render(phase):
        return visualizer.render_state(
            env.state, mdp.terrain_mtx,
            hud_data=visualizer.default_hud_data(
                env.state, score=score, time_left=400 - env.state.timestep,
                phase=phase, handoffs=successes if sync else "ordinary counter",
            ),
        )

    def video_frame(surface):
        rgb = pygame.surfarray.array3d(surface).transpose(1, 0, 2)
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        height, width = frame.shape[:2]
        return cv2.copyMakeBorder(frame, 0, height % 2, 0, width % 2, cv2.BORDER_REPLICATE)

    with tempfile.TemporaryDirectory(prefix="hucooked-three-player-") as temp_dir:
        temp_video = Path(temp_dir) / "demo.mp4"
        initial = render("Three chefs: supply, cook, serve")
        pygame.image.save(initial, str(output / "initial_state.png"))
        first_frame = video_frame(initial)
        height, width = first_frame.shape[:2]
        writer = cv2.VideoWriter(str(temp_video), cv2.VideoWriter_fourcc(*"mp4v"), 6, (width, height))
        if not writer.isOpened():
            raise RuntimeError("Could not open the video encoder")
        writer.write(first_frame)

        def step(actions, phase):
            nonlocal score, attempts, successes, wait_agent_steps
            state, reward, _, info = env.step(actions)
            score += reward
            attempts += len(info["sync_handoffs"])
            successes += sum(event["success"] for event in info["sync_handoffs"])
            wait_agent_steps += sum(info["event_infos"]["sync_handoff_wait"])
            mdp._check_valid_state(state)
            trajectory.append({
                "step": state.timestep, "phase": phase,
                "actions": [Action.ACTION_TO_INDEX[a] for a in actions],
                "reward": reward, "state": state.to_dict(),
                "sync_handoffs": info["sync_handoffs"],
            })
            surface = render(phase)
            writer.write(video_frame(surface))
            if phase == "Only A presses: transfer must fail":
                pygame.image.save(surface, str(output / "failed_handoff.png"))
            if sync and successes == 1 and info["sync_handoffs"]:
                pygame.image.save(surface, str(output / "successful_handoff.png"))
            return info

        def walk(agent, target, facing, phase):
            """Move one chef at a time via BFS; never teleport or edit state."""
            player = env.state.players[agent]
            blocked = {p.position for i, p in enumerate(env.state.players) if i != agent}
            queue = deque([(player.position, [])])
            seen = {player.position}
            path = None
            while queue:
                position, actions = queue.popleft()
                if position == target:
                    path = actions
                    break
                for direction in Direction.ALL_DIRECTIONS:
                    neighbor = Action.move_in_direction(position, direction)
                    if neighbor in mdp.get_valid_player_positions() and neighbor not in blocked and neighbor not in seen:
                        seen.add(neighbor)
                        queue.append((neighbor, actions + [direction]))
            if path is None:
                raise RuntimeError("No path for chef {} to {}".format(agent, target))
            for action in path:
                joint = [W, W, W]
                joint[agent] = action
                step(joint, phase)
            if env.state.players[agent].orientation != facing:
                # All targets face non-walkable facilities, so this turns in place.
                joint = [W, W, W]
                joint[agent] = facing
                step(joint, phase)
            assert env.state.players[agent].pos_and_or == (target, facing)

        try:
            walk(2, (9, 4), Direction.EAST, "C prepares to collect a dish")
            for ingredient in range(3):
                walk(1, (6, 3), Direction.WEST, "B arrives at the handoff station")
                walk(0, (1, 1), Direction.WEST, "A goes to the onion supply")
                step((I, W, W), "A picks up an onion")
                walk(0, (4, 3), Direction.EAST, "A brings an onion to the station")
                if sync:
                    if ingredient == 0:
                        info = step((I, W, W), "Only A presses: transfer must fail")
                        assert not info["sync_handoffs"][0]["success"]
                        assert env.state.players[0].has_object()
                        info = step((W, I, W), "Only B presses later: transfer still fails")
                        assert not info["sync_handoffs"][0]["success"]
                    step((I, I, I if ingredient == 0 else W), "A and B press together; C can work too")
                else:
                    step((I, W, I if ingredient == 0 else W), "A leaves the onion on an ordinary counter")
                    step((W, W, W), "The onion can wait on this counter")
                    step((W, I, W), "B picks it up on a later step")
                assert not env.state.players[0].has_object()
                assert env.state.players[1].get_object().name == "onion"
                assert env.state.players[2].get_object().name == "dish"
                walk(1, (8, 1), Direction.NORTH, "B takes the onion to the pot")
                step((W, I, W), "B adds onion {} of 3".format(ingredient + 1))

            step((W, I, W), "B starts cooking")
            walk(1, (7, 2), Direction.WEST, "B clears the way for C")
            walk(2, (8, 1), Direction.NORTH, "C brings the dish to the pot")
            while not env.state.get_object((8, 0)).is_ready:
                step((W, W, W), "The soup is cooking")
            step((W, W, I), "C plates the cooked soup")
            walk(2, (8, 5), Direction.SOUTH, "C carries the soup to the serving point")
            step((W, W, I), "Soup delivered: team earns 20 points")
            assert score == 20
            if sync:
                assert successes == 3 and attempts == 5
            final = render("Complete: three chefs made one soup")
            pygame.image.save(final, str(output / "final_state.png"))
            for _ in range(6):
                writer.write(video_frame(final))
        finally:
            writer.release()

        capture = cv2.VideoCapture(str(temp_video))
        try:
            readable, _ = capture.read()
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        assert readable and frames == env.state.timestep + 7
        shutil.copy2(temp_video, output / "demo.mp4")

    summary = {
        "layout": args.layout, "players": 3, "steps": env.state.timestep,
        "team_score": score, "soups_delivered": 1,
        "observation_shape_per_player": list(observations[0].shape),
        "sync_station_attempts": attempts if sync else None,
        "sync_station_successes": successes if sync else None,
        "sync_wait_agent_steps": wait_agent_steps if sync else None,
        "video_frames": frames,
        "controller": "Scripted rule check, not a trained RL policy",
        "note": (
            "Two failed presses are deliberate test cases; these metrics do not measure policy quality."
            if sync else "Ordinary counter rule check; these metrics do not measure policy quality."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
    pygame.quit()
    print(json.dumps(summary, indent=2))
    print("Output directory:", output)


if __name__ == "__main__":
    main()
