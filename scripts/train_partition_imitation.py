"""Train a shared behavior-cloning MLP and evaluate it on held-out paired starts."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from evaluate_partition_kitchen import evaluate
from overcooked_ai_py.cooperative.partition import PartitionConfig, PartitionKitchen
from overcooked_ai_py.cooperative.partition.controller import ScriptedChefs


class ImitationPolicy:
    def __init__(self, env, model):
        self.env = env
        self.model = model

    @torch.inference_mode()
    def actions(self):
        obs = torch.from_numpy(self.env.observations())
        return self.model(obs).argmax(dim=1).tolist()


def demonstrations(episodes, first_seed):
    observations, labels = [], []
    for layout in ("sync", "counter"):
        env = PartitionKitchen(PartitionConfig(layout=layout))
        for seed in range(first_seed, first_seed + episodes):
            obs, _ = env.reset(seed)
            team = ScriptedChefs(env)
            for _ in range(env.config.horizon):
                action = team.actions()
                observations.append(obs)
                labels.append(action)
                obs, _, _, _, _ = env.step(action)
    return np.concatenate(observations), np.asarray(labels).ravel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-episodes", type=int, default=32, help="Per layout, starting at seed 1000")
    parser.add_argument("--eval-episodes", type=int, default=30, help="Per layout, starting at seed 0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("demo_outputs/partition"))
    args = parser.parse_args()
    if min(args.train_episodes, args.eval_episodes, args.epochs) < 1 or args.eval_episodes > 1000:
        parser.error("Use positive counts and at most 1000 held-out episodes per layout")
    torch.manual_seed(0)
    torch.set_num_threads(1)
    x, y = demonstrations(args.train_episodes, 1000)
    x = torch.as_tensor(x, device=args.device)
    y = torch.as_tensor(y, dtype=torch.long, device=args.device)
    model = nn.Sequential(nn.Linear(x.shape[1], 128), nn.ReLU(), nn.Linear(128, 128),
                          nn.ReLU(), nn.Linear(128, 6)).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(args.epochs):
        for idx in torch.randperm(len(y), device=args.device).split(512):
            optimizer.zero_grad()
            loss = loss_fn(model(x[idx]), y[idx])
            loss.backward()
            optimizer.step()
    model.cpu().eval()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output / "imitation.pt")
    results = evaluate(args.eval_episodes, 0, policy_factory=lambda env: ImitationPolicy(env, model))
    report = dict(method="behavior_cloning", seed=0, train_episodes_per_map=args.train_episodes,
                  demonstration_actions=len(y), epochs=args.epochs, eval_episodes_per_map=args.eval_episodes,
                  results=results)
    (args.output / "imitation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
