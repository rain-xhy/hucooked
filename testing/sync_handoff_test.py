"""Behavioral checks for the three-player kitchen and atomic H handoffs."""

import unittest
from collections import deque

import numpy as np

from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import (
    ObjectState,
    OvercookedGridworld,
    OvercookedState,
    PlayerState,
)
from overcooked_ai_py.utils import read_layout_dict
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

I, W = Action.INTERACT, Action.STAY


class TestSyncHandoff(unittest.TestCase):
    def setUp(self):
        self.mdp = OvercookedGridworld.from_layout_name("three_player_sync_handoff")

    def ready_state(self, left="onion", right=None):
        state = self.mdp.get_standard_start_state()
        specs = [((4, 3), Direction.EAST, left),
                 ((6, 3), Direction.WEST, right),
                 ((9, 4), Direction.EAST, None)]
        state.players = tuple(
            PlayerState(pos, direction, ObjectState(obj, pos) if obj else None)
            for pos, direction, obj in specs
        )
        return state

    def test_three_player_step_and_episode_end(self):
        env = OvercookedEnv.from_mdp(self.mdp, horizon=2)
        state, reward, done, info = env.step((W, W, W))
        self.assertEqual(len(state.players), 3)
        self.assertEqual(info["agent_infos"], [{}, {}, {}])
        self.assertEqual(len(info["sparse_r_by_agent"]), 3)
        self.assertEqual(reward, 0)
        self.assertFalse(done)
        _, _, done, info = env.step((W, W, W))
        self.assertTrue(done)
        self.assertEqual(info["episode"]["ep_length"], 2)

    def test_wrong_number_of_actions_does_not_advance_state(self):
        env = OvercookedEnv.from_mdp(self.mdp)
        before = env.state.deepcopy()
        for actions in ((W, W), (W, W, W, W)):
            with self.assertRaises(ValueError):
                env.step(actions)
            self.assertEqual(env.state, before)
        with self.assertRaises(ValueError):
            env.step((W, W, W), [{}, {}])
        self.assertEqual(env.state, before)

    def test_atomic_transfer_and_independent_third_player(self):
        state = self.ready_state()
        original = state.deepcopy()
        after, info = self.mdp.get_state_transition(state, (I, I, I))
        self.assertEqual(state, original)
        self.assertFalse(after.players[0].has_object())
        self.assertEqual(after.players[1].get_object().name, "onion")
        self.assertEqual(after.players[1].get_object().position, (6, 3))
        self.assertEqual(after.players[2].get_object().name, "dish")
        self.assertFalse(after.has_object((5, 3)))
        self.assertEqual(info["sparse_reward_by_agent"], [0, 0, 0])
        self.assertEqual(info["shaped_reward_by_agent"][:2], [0, 0])
        self.assertEqual(info["event_infos"]["sync_handoff_success"], [True, True, False])
        self.assertEqual(len(info["sync_handoffs"]), 1)
        self.assertTrue(info["sync_handoffs"][0]["success"])

    def test_player_number_does_not_determine_transfer(self):
        original = self.ready_state()
        after, _ = self.mdp.get_state_transition(original, (I, I, W))
        permuted = self.ready_state()
        permuted.players = tuple(permuted.players[i] for i in (2, 1, 0))
        other, _ = self.mdp.get_state_transition(permuted, (W, I, I))
        by_pos = lambda s: {p.position: p.held_object for p in s.players}
        self.assertEqual(by_pos(after), by_pos(other))

    def test_transfer_can_go_in_either_direction(self):
        state = self.ready_state(left=None, right="tomato")
        after, info = self.mdp.get_state_transition(state, (I, I, W))
        self.assertEqual(after.players[0].get_object().name, "tomato")
        self.assertFalse(after.players[1].has_object())
        self.assertEqual(info["sync_handoffs"][0]["sender"], 1)

    def test_staggered_actions_never_transfer(self):
        state = self.ready_state()
        for actions in ((I, W, W), (W, I, W)):
            state, info = self.mdp.get_state_transition(state, actions)
            self.assertTrue(state.players[0].has_object())
            self.assertFalse(state.players[1].has_object())
            self.assertFalse(state.has_object((5, 3)))
            self.assertEqual(info["sync_handoffs"][0]["reason"], "not_simultaneous")

    def test_both_empty_or_both_holding_cannot_transfer(self):
        for left, right, reason in ((None, None, "both_empty"),
                                    ("onion", "dish", "both_holding")):
            with self.subTest(reason=reason):
                state = self.ready_state(left, right)
                after, info = self.mdp.get_state_transition(state, (I, I, W))
                self.assertTrue(state.time_independent_equal(after))
                self.assertEqual(info["sync_handoffs"][0]["reason"], reason)

    def test_wrong_orientation_cannot_transfer(self):
        state = self.ready_state()
        state.players[1].orientation = Direction.EAST
        after, info = self.mdp.get_state_transition(state, (I, I, W))
        self.assertTrue(after.players[0].has_object())
        self.assertFalse(after.players[1].has_object())
        self.assertEqual(info["sync_handoffs"][0]["reason"], "waiting_for_partner")

    def test_arriving_player_must_wait_until_next_step_to_interact(self):
        state = self.ready_state()
        state.players[1].update_pos_and_or((7, 3), Direction.WEST)
        state, info = self.mdp.get_state_transition(state, (I, Direction.WEST, W))
        self.assertTrue(state.players[0].has_object())
        self.assertFalse(state.players[1].has_object())
        self.assertEqual(state.players[1].position, (6, 3))
        state, info = self.mdp.get_state_transition(state, (I, I, W))
        self.assertTrue(info["sync_handoffs"][0]["success"])

    def test_handoff_station_is_not_walkable(self):
        state = self.ready_state()
        after, _ = self.mdp.get_state_transition(
            state, (Direction.EAST, Direction.WEST, W)
        )
        self.assertEqual(after.player_positions, state.player_positions)

    def test_repeated_handoffs_pay_no_reward_and_preserve_one_object(self):
        state = self.ready_state()
        for _ in range(6):
            state, info = self.mdp.get_state_transition(state, (I, I, W))
            self.assertEqual(sum(p.has_object() for p in state.players), 1)
            self.assertEqual(sum(info["sparse_reward_by_agent"]), 0)
            self.assertEqual(sum(info["shaped_reward_by_agent"]), 0)

    def test_event_metrics_count_one_station_attempt_and_two_waiting_agents(self):
        env = OvercookedEnv.from_mdp(self.mdp, start_state_fn=self.ready_state)
        _, _, _, info = env.step((I, W, W))
        self.assertEqual(len(info["sync_handoffs"]), 1)
        self.assertEqual(info["event_infos"]["sync_handoff_wait"], [True, True, False])
        _, _, _, info = env.step((I, I, W))
        self.assertEqual(len(info["sync_handoffs"]), 1)
        self.assertEqual(info["event_infos"]["sync_handoff_wait"], [False, False, False])
        self.assertEqual(env.game_stats["sync_handoff_give"][0], [1])
        self.assertEqual(env.game_stats["sync_handoff_success"][1], [1])

    def test_observations_include_all_players_and_handoff_station(self):
        state = self.mdp.get_standard_start_state()
        observations = self.mdp.lossless_state_encoding(state)
        self.assertEqual(len(observations), 3)
        self.assertEqual(tuple(self.mdp.get_lossless_state_encoding_shape()), (11, 7, 32))
        for i, obs in enumerate(observations):
            self.assertEqual(obs.shape, (11, 7, 32))
            self.assertEqual(obs[:, :, 0].sum(), 1)
            self.assertEqual(obs[state.players[i].position + (0,)], 1)
            # Five channels per player, then six ordinary terrain planes and H.
            self.assertEqual(obs[5, 3, 21], 1)
            self.assertEqual(obs[:, :, 21].sum(), 1)
            self.assertEqual(obs[:, :, :3].sum(), 3)

    def test_old_two_player_encoding_is_unchanged(self):
        mdp = OvercookedGridworld.from_layout_name("cramped_room")
        obs = mdp.lossless_state_encoding(mdp.get_standard_start_state())
        self.assertEqual(len(obs), 2)
        self.assertEqual(obs[0].shape, (5, 4, 26))

    def test_ordinary_three_player_counter_allows_delayed_pickup(self):
        mdp = OvercookedGridworld.from_layout_name("three_player_handoff")
        state = self.ready_state()
        state, _ = mdp.get_state_transition(state, (I, W, W))
        self.assertTrue(state.has_object((5, 3)))
        state, _ = mdp.get_state_transition(state, (W, W, W))
        state, _ = mdp.get_state_transition(state, (W, I, W))
        self.assertEqual(state.players[1].get_object().name, "onion")
        self.assertFalse(state.has_object((5, 3)))
        self.assertEqual(mdp.lossless_state_encoding(state)[0].shape, (11, 7, 31))

    def test_state_round_trip_and_mdp_recreation(self):
        state, _ = self.mdp.get_state_transition(self.ready_state(), (I, I, W))
        self.assertEqual(OvercookedState.from_dict(state.to_dict()), state)
        for mdp in (self.mdp.copy(), OvercookedGridworld(**self.mdp.mdp_params)):
            self.assertEqual(mdp.get_sync_handoff_locations(), [(5, 3)])
            self.assertEqual(mdp.num_players, 3)

    def test_invalid_handoff_geometry_is_rejected(self):
        grid = read_layout_dict("three_player_sync_handoff")["grid"]
        rows = [list(row.strip()) for row in grid.splitlines()]
        rows[2][5] = " "
        with self.assertRaisesRegex(AssertionError, "two opposite"):
            OvercookedGridworld.from_grid(rows)

    def test_ingredients_cannot_bypass_the_handoff_station(self):
        def reachable(start):
            queue, visited = deque([start]), {start}
            floor = set(self.mdp.get_valid_player_positions())
            while queue:
                position = queue.popleft()
                for direction in Direction.ALL_DIRECTIONS:
                    neighbor = Action.move_in_direction(position, direction)
                    if neighbor in floor and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            return visited

        left = reachable(self.mdp.start_player_positions[0])
        right = reachable(self.mdp.start_player_positions[1])
        self.assertTrue(left.isdisjoint(right))
        self.assertIn(self.mdp.start_player_positions[2], right)
        for counter in self.mdp.get_counter_locations():
            neighbors = {
                Action.move_in_direction(counter, d) for d in Direction.ALL_DIRECTIONS
            }
            self.assertFalse(neighbors & left and neighbors & right)
        self.assertEqual(self.mdp.get_sync_handoff_locations(), [(5, 3)])

    def test_handoff_station_rejects_stored_objects(self):
        state = self.ready_state()
        state.add_object(ObjectState("dish", (5, 3)))
        with self.assertRaisesRegex(AssertionError, "cannot store"):
            self.mdp.get_state_transition(state, (I, I, W))

    def test_vertical_station_supports_same_rule(self):
        grid = read_layout_dict("three_player_sync_handoff")["grid"]
        rows = [row.strip() for row in grid.splitlines()]
        mdp = OvercookedGridworld.from_grid(["".join(col) for col in zip(*rows)])
        state = mdp.get_standard_start_state()
        state.players = (
            PlayerState((3, 4), Direction.SOUTH, ObjectState("onion", (3, 4))),
            PlayerState((3, 6), Direction.NORTH),
            PlayerState((4, 9), Direction.SOUTH),
        )
        after, info = mdp.get_state_transition(state, (I, I, W))
        self.assertTrue(info["sync_handoffs"][0]["success"])
        self.assertEqual(after.players[1].get_object().name, "onion")

    def test_random_three_player_episode_and_rendering(self):
        rng = np.random.default_rng(0)
        env = OvercookedEnv.from_mdp(self.mdp, horizon=100)
        for _ in range(100):
            actions = tuple(Action.INDEX_TO_ACTION[i] for i in rng.integers(0, 6, 3))
            state, _, done, _ = env.step(actions)
            self.mdp._check_valid_state(state)
        self.assertTrue(done)
        surface = StateVisualizer().render_state(state, self.mdp.terrain_mtx)
        self.assertEqual(surface.get_size(), (825, 525))

    def test_unsupported_legacy_planner_fails_with_explanation(self):
        env = OvercookedEnv.from_mdp(self.mdp)
        with self.assertRaisesRegex(NotImplementedError, "legacy"):
            _ = env.mlam


if __name__ == "__main__":
    unittest.main()
