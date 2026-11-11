"""Render a complete standard shift; outputs stay local under demo_outputs."""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import cv2

from overcooked_ai_py.cooperative import MixedKitchen
from overcooked_ai_py.cooperative.controller import ReferenceController


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("demo_outputs/mixed_kitchen/demo"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env, policy = MixedKitchen(), ReferenceController()
    env.reset(args.seed)
    trajectory = []

    def frame():
        rgb = env.render()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        return cv2.copyMakeBorder(bgr, 0, h % 2, 0, w % 2, cv2.BORDER_REPLICATE)

    def save_image(name, image):
        # imencode + tofile supports Windows paths containing Chinese characters.
        ok, data = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("Could not encode the demonstration image")
        data.tofile(str(args.output_dir / name))

    with tempfile.TemporaryDirectory(prefix="mixed-kitchen-") as temp:
        video = Path(temp) / "demo.mp4"
        first = frame()
        save_image("initial_state.png", first)
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 8,
                                 (first.shape[1], first.shape[0]))
        if not writer.isOpened():
            raise RuntimeError("Could not open MP4 encoder")
        saved_parallel = False
        try:
            writer.write(first)
            for _ in range(env.config.horizon):
                actions = policy(env)
                _, _, done, _, info = env.step(actions)
                trajectory.append({"actions": actions, "state": env.state.to_dict(),
                                   "events": info["events"], "raw_reward": info["raw_reward"]})
                image = frame()
                writer.write(image)
                if not saved_parallel and env.state.joint.progress and env.state.orders[0].status == "active":
                    save_image("parallel_work.png", image)
                    saved_parallel = True
                if done:
                    break
            save_image("final_state.png", image)
        finally:
            writer.release()
        capture = cv2.VideoCapture(str(video))
        readable, _ = capture.read()
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if not readable or count != env.state.tick + 1:
            raise RuntimeError("Exported video is incomplete")
        shutil.copy2(video, args.output_dir / "demo.mp4")
    summary = {"seed": args.seed, "fingerprint": env.fingerprint,
               "config": env.config.to_dict(), "score": env.state.score,
               "steps": env.state.tick, "metrics": env.state.metrics,
               "controller": "Public-state scripted reference; not a trained policy"}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Output:", args.output_dir)


if __name__ == "__main__":
    main()
