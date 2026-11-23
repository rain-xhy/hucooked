import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from . import PartitionKitchen


class ParallelPartitionKitchen(ParallelEnv):
    metadata = {"name": "partition_kitchen_v1", "render_modes": [], "is_parallelizable": True}

    def __init__(self, config=None):
        self.core = PartitionKitchen(config)
        self.possible_agents = [f"chef_{i}" for i in range(3)]
        self.agents = self.possible_agents.copy()
        self.action_spaces = {a: spaces.Discrete(6) for a in self.agents}
        shape = self.core.observations()[0].shape
        self.observation_spaces = {a: spaces.Box(0, 1, shape, np.float32) for a in self.agents}
        self.state_space = spaces.Box(0, 1, self.core.centralized_state().shape, np.float32)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def state(self):
        return self.core.centralized_state()

    def reset(self, seed=None, options=None):
        obs, _ = self.core.reset(seed)
        self.agents = self.possible_agents.copy()
        for i, agent in enumerate(self.agents):
            self.action_spaces[agent].seed(None if seed is None else seed + i)
        return dict(zip(self.agents, obs)), {a: {} for a in self.agents}

    def step(self, actions):
        if set(actions) != set(self.agents):
            raise ValueError("Expected one action per live chef")
        if not self.agents:
            return {}, {}, {}, {}, {}
        agents = self.agents.copy()
        obs, reward, term, trunc, info = self.core.step([actions[a] for a in agents])
        if term or trunc:
            self.agents = []
        return (dict(zip(agents, obs)), dict.fromkeys(agents, reward), dict.fromkeys(agents, term),
                dict.fromkeys(agents, trunc), {a: {k: v.copy() for k, v in info.items()} for a in agents})
