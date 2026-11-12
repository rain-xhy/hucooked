"""Parity checks for the tensor backend using the native MDP as the oracle."""

import importlib.util
import os
from pathlib import Path
import random
import unittest
from unittest.mock import patch

import numpy as np

from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import (
    ObjectState, OvercookedGridworld, Recipe, SoupState,
)
from overcooked_ai_py.static import LAYOUTS_DIR


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch dependency")
class TorchOvercookedTest(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch
        self.device = os.getenv("OVERCOOKED_TEST_DEVICE", "cpu")
        torch.set_num_threads(1)

    def assert_state(self, actual, expected):
        self.assertEqual(actual.timestep, expected.timestep)
        self.assertEqual(actual.all_orders, expected.all_orders)
        self.assertEqual(actual.bonus_orders, expected.bonus_orders)
        self.assertEqual([p.to_dict() for p in actual.players], [p.to_dict() for p in expected.players])
        self.assertEqual({p: x.to_dict() for p, x in actual.objects.items()},
                         {p: x.to_dict() for p, x in expected.objects.items()})

    def test_every_layout_matches_native_transitions(self):
        rng = random.Random(172)
        for layout in sorted(Path(LAYOUTS_DIR).glob("*.layout")):
            with self.subTest(layout=layout.stem):
                mdp = OvercookedGridworld.from_layout_name(layout.stem)
                env = OvercookedEnv.from_mdp(mdp, horizon=25, device=self.device, num_envs=2)
                states = [mdp.get_standard_start_state()] * 2
                for tick in range(25):
                    joint = [[rng.randrange(6) for _ in range(mdp.num_players)] for _ in range(2)]
                    actions = self.torch.tensor(joint, device=self.device)
                    tensor_state, reward, done, info = env.step(actions)
                    self.assertEqual(tensor_state.timestep.device.type, self.device)
                    for lane, ids in enumerate(joint):
                        states[lane], expected = mdp.get_state_transition(
                            states[lane], tuple(Action.INDEX_TO_ACTION[i] for i in ids),
                        )
                        self.assert_state(env.to_native_state(lane), states[lane])
                        self.assertEqual(reward[lane].item(), sum(expected["sparse_reward_by_agent"]))
                        self.assertEqual(info["shaped_r_by_agent"][lane].tolist(), expected["shaped_reward_by_agent"])
                        for event, flags in expected["event_infos"].items():
                            self.assertEqual(info["events"][event][lane].tolist(), flags)
                    self.assertEqual(done.tolist(), [tick == 24] * 2)
                if mdp.num_players == 2:
                    grid = env.lossless_state_encoding()
                    for lane, state in enumerate(states):
                        reference = self.torch.as_tensor(
                            np.stack(mdp.lossless_state_encoding(state, horizon=25)),
                            dtype=self.torch.long, device=self.device,
                        )
                        self.torch.testing.assert_close(grid[lane], reference)

    def test_mixed_soup_is_plated_and_bonus_reward_is_served(self):
        mdp = OvercookedGridworld.from_layout_name("cramped_room_tomato")
        state = mdp.get_standard_start_state()
        pot = mdp.get_pot_locations()[0]
        state.add_object(SoupState.get_soup(pot, num_onions=1, num_tomatoes=1, finished=True))
        state.players[0].update_pos_and_or((2, 1), Direction.NORTH)
        state.players[0].set_object(ObjectState("dish", (2, 1)))
        state.players[1].update_pos_and_or((1, 1), Direction.WEST)
        env = OvercookedEnv.from_mdp(mdp, horizon=5, device=self.device, num_envs=2,
                                    start_state_fn=lambda: state)
        actions = [5, 1, 2, 1, 5]  # Scoop, south, east, face the serving station, deliver.
        reference = state
        for a in actions:
            _, reward, done, info = env.step(self.torch.tensor([[a, 4], [a, 4]], device=self.device))
            reference, expected = mdp.get_state_transition(reference, (Action.INDEX_TO_ACTION[a], Action.STAY))
            self.assert_state(env.to_native_state(0), reference)
            self.assertEqual(reward[0].item(), sum(expected["sparse_reward_by_agent"]))
        self.assertEqual(reward.tolist(), [50, 50])
        self.assertEqual(info["events"]["soup_delivery"].tolist(), [[True, False]] * 2)
        self.assertTrue(done.all())
        env.reset(mask=self.torch.tensor([True, False], device=self.device))
        self.assert_state(env.to_native_state(0), state)
        self.assert_state(env.to_native_state(1), reference)

    def test_old_dynamics_starts_full_pot_automatically(self):
        with patch.object(Recipe, "_configured", False):
            mdp = OvercookedGridworld.from_layout_name("old_dynamics_put_test", old_dynamics=True)
            native, _ = mdp.get_state_transition(mdp.get_standard_start_state(), (Action.INTERACT,))
            env = OvercookedEnv.from_mdp(mdp, device=self.device)
            env.step(self.torch.tensor([[5]], device=self.device))
            self.assert_state(env.to_native_state(0), native)
            self.assertEqual(native.objects[mdp.get_pot_locations()[0]]._cooking_tick, 1)
