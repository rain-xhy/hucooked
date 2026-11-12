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
    ObjectState, OvercookedGridworld, PlayerState, Recipe, SoupState,
)
from overcooked_ai_py.static import LAYOUTS_DIR
from overcooked_ai_py.utils import read_layout_dict


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch dependency")
class TorchOvercookedTest(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch
        requested_device = torch.device(os.getenv("OVERCOOKED_TEST_DEVICE", "cpu"))
        self.device = torch.empty(0, device=requested_device).device
        torch.set_num_threads(1)

    def assert_state(self, actual, expected):
        self.assertEqual(actual.timestep, expected.timestep)
        self.assertEqual(actual.all_orders, expected.all_orders)
        self.assertEqual(actual.bonus_orders, expected.bonus_orders)
        self.assertEqual([p.to_dict() for p in actual.players], [p.to_dict() for p in expected.players])
        self.assertEqual({p: x.to_dict() for p, x in actual.objects.items()},
                         {p: x.to_dict() for p, x in expected.objects.items()})

    def assert_observations(self, env, states):
        grid = env.lossless_state_encoding()
        self.assertEqual(grid.device, self.device)
        for lane, state in enumerate(states):
            reference = self.torch.as_tensor(
                np.stack(env.mdp.lossless_state_encoding(state, horizon=env.horizon)),
                dtype=self.torch.long, device=self.device,
            )
            self.torch.testing.assert_close(grid[lane], reference)

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
                    self.assertEqual(tensor_state.timestep.device, self.device)
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
                self.assert_observations(env, states)

    def test_batched_sync_handoffs_match_native(self):
        mdp = OvercookedGridworld.from_layout_name("three_player_sync_handoff")
        positions = ((4, 3), (6, 3), (9, 4))
        interact, stay = Action.INTERACT, Action.STAY
        cases = [
            ("onion", None, Direction.WEST, (interact, interact, interact)),
            (None, "tomato", Direction.WEST, (interact, interact, stay)),
            ("soup", None, Direction.WEST, (interact, interact, stay)),
            ("onion", None, Direction.WEST, (interact, stay, stay)),
            (None, None, Direction.WEST, (interact, interact, stay)),
            ("onion", "dish", Direction.WEST, (interact, interact, stay)),
            ("onion", None, Direction.EAST, (interact, interact, stay)),
            ("onion", None, Direction.WEST, (stay, stay, stay)),
            ("onion", None, Direction.WEST, (Direction.EAST, Direction.WEST, stay)),
        ]
        starts = []
        for left, right, facing, _ in cases:
            state = mdp.get_standard_start_state()
            state.players = tuple(
                PlayerState(
                    pos, direction,
                    SoupState.get_soup(pos, num_onions=1, num_tomatoes=1, finished=True)
                    if name == "soup" else ObjectState(name, pos) if name else None,
                )
                for pos, direction, name in zip(positions, (Direction.EAST, facing, Direction.EAST),
                                                (left, right, None))
            )
            starts.append(state)
        start_iter = iter(starts)
        env = OvercookedEnv.from_mdp(mdp, horizon=1, device=self.device, num_envs=len(starts),
                                    start_state_fn=lambda: next(start_iter))
        actions = [case[3] for case in cases]
        indices = self.torch.tensor(
            [[Action.ACTION_TO_INDEX[action] for action in joint] for joint in actions], device=self.device,
        )
        _, reward, done, info = env.step(indices)
        self.assertTrue(done.all())
        after = []
        for lane, (before, joint) in enumerate(zip(starts, actions)):
            state, native_info = mdp.get_state_transition(before, joint)
            after.append(state)
            self.assert_state(env.to_native_state(lane), state)
            self.assertFalse(state.has_object((5, 3)))
            self.assertEqual(reward[lane].item(), sum(native_info["sparse_reward_by_agent"]))
            self.assertEqual(info["shaped_r_by_agent"][lane].tolist(), native_info["shaped_reward_by_agent"])
            for event, expected in native_info["event_infos"].items():
                self.assertEqual(info["events"][event][lane].tolist(), expected)
            self.assertEqual(info["sync_handoffs"].to_native(lane), native_info["sync_handoffs"])
        self.assert_observations(env, after)
        _, reward, _, info = env.step(indices)
        self.assertFalse(reward.any())
        for lane, state in enumerate(after):
            self.assert_state(env.to_native_state(lane), state)
            self.assertEqual(info["sync_handoffs"].to_native(lane), [])
        self.assertFalse(info["events"]["sync_handoff_wait"].any())

    def test_vertical_sync_handoff_and_multi_player_observation_order(self):
        grid = read_layout_dict("three_player_sync_handoff")["grid"]
        rows = [row.strip() for row in grid.splitlines()]
        mdp = OvercookedGridworld.from_grid(["".join(column) for column in zip(*rows)])
        start = mdp.get_standard_start_state()
        start.players = (
            PlayerState((4, 9), Direction.SOUTH),
            PlayerState((3, 6), Direction.NORTH),
            PlayerState((3, 4), Direction.SOUTH, ObjectState("dish", (3, 4))),
        )
        env = OvercookedEnv.from_mdp(mdp, device=self.device, start_state_fn=lambda: start)
        joint = (Action.STAY, Action.INTERACT, Action.INTERACT)
        _, _, _, info = env.step([[Action.ACTION_TO_INDEX[action] for action in joint]])
        after, native_info = mdp.get_state_transition(start, joint)
        self.assert_state(env.to_native_state(), after)
        self.assertEqual(info["sync_handoffs"].to_native(0), native_info["sync_handoffs"])
        self.assert_observations(env, [after])

    def test_custom_start_cannot_store_items_on_handoff_station(self):
        mdp = OvercookedGridworld.from_layout_name("three_player_sync_handoff")
        start = mdp.get_standard_start_state()
        station, = mdp.get_sync_handoff_locations()
        start.add_object(ObjectState("dish", station))
        with self.assertRaisesRegex(AssertionError, "cannot store"):
            OvercookedEnv.from_mdp(mdp, device=self.device, start_state_fn=lambda: start)

    def test_two_stations_transfer_independent_items_in_the_same_step(self):
        mdp = OvercookedGridworld.from_grid([
            "XXXXXXXXXXX",
            "X O D P S X",
            "X  X   X  X",
            "X 1H2 3H4 X",
            "X  X   X  X",
            "XXXXXXXXXXX",
        ])
        start = mdp.get_standard_start_state()
        for i, player in enumerate(start.players):
            player.orientation = Direction.EAST if i % 2 == 0 else Direction.WEST
        start.players[0].set_object(ObjectState("onion", start.players[0].position))
        start.players[3].set_object(ObjectState("dish", start.players[3].position))
        env = OvercookedEnv.from_mdp(mdp, device=self.device, start_state_fn=lambda: start)
        joint = (Action.INTERACT,) * mdp.num_players
        _, _, _, info = env.step([[Action.ACTION_TO_INDEX[Action.INTERACT]] * mdp.num_players])
        after, native_info = mdp.get_state_transition(start, joint)
        self.assert_state(env.to_native_state(), after)
        self.assertEqual(info["sync_handoffs"].to_native(0), native_info["sync_handoffs"])
        self.assert_observations(env, [after])

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
