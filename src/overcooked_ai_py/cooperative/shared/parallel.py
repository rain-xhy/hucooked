import copy
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from .kitchen import STAY, SharedKitchen


class ParallelSharedKitchen(ParallelEnv):
    metadata = {"name": "shared_kitchen_v1", "render_modes": ["rgb_array"], "is_parallelizable": True}

    def __init__(self, config=None, render_mode=None):
        if render_mode not in (None, "rgb_array"):
            raise ValueError("Only rgb_array rendering is supported")
        self.core = SharedKitchen(config)
        self.render_mode = render_mode
        self.possible_agents = ["chef_" + str(i) for i in self.core.active]
        self.agents = self.possible_agents.copy()
        shape = self.core.observations()[0].shape
        self.observation_spaces = {agent: spaces.Box(0.0, 1.0, shape, dtype=np.float32) for agent in self.possible_agents}
        self.action_spaces = {agent: spaces.Discrete(6) for agent in self.possible_agents}
        self.state_space = spaces.Box(0.0, 1.0, self.core.centralized_state().shape, dtype=np.float32)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        if options is not None and not isinstance(options, dict):
            raise TypeError("options must be a dictionary")
        observations, info = self.core.reset(seed)
        self.agents = self.possible_agents.copy()
        for i, agent in enumerate(self.possible_agents):
            self.action_spaces[agent].seed(None if seed is None else seed + i)
        return ({agent: observations[int(agent[-1])] for agent in self.agents}, {agent: copy.deepcopy(info) for agent in self.agents})

    def step(self, actions):
        if not self.agents:
            if actions:
                raise RuntimeError("Reset before stepping a finished episode")
            return {}, {}, {}, {}, {}
        if set(actions) != set(self.agents):
            raise ValueError("One action per active chef is required")
        joint = [STAY] * 3
        for agent, action in actions.items():
            joint[int(agent[-1])] = action
        observations, reward, terminated, truncated, info = self.core.step(joint)
        live = self.agents.copy()
        result = ({agent: observations[int(agent[-1])] for agent in live},
                  {agent: reward for agent in live},
                  {agent: terminated for agent in live},
                  {agent: truncated for agent in live},
                  {agent: copy.deepcopy(info) for agent in live})
        if terminated or truncated:
            self.agents = []
        return result

    def state(self):
        return self.core.centralized_state()

    def render(self):
        return self.core.render()

    def close(self):
        pass
