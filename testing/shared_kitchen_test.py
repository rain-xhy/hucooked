import copy
import io
import itertools
import json
from pathlib import Path
import tokenize
import unittest
import numpy as np
from overcooked_ai_py.cooperative.shared import EAST, INTERACT, NORTH, SOUTH, STAY, WEST, SharedKitchen, SharedKitchenConfig
from overcooked_ai_py.cooperative.shared.controller import ScriptedController, navigation_action
from overcooked_ai_py.cooperative.shared.layouts import COOKER_PORTS, FLOOR, LAYOUT, STATIONS
from overcooked_ai_py.cooperative.shared.model import Ticket


def act(env, chef_id, action):
    actions = [STAY] * 3
    actions[chef_id] = action
    return env.step(actions)


def interact_at(env, chef_id, position, station):
    for _ in range(100):
        action = navigation_action(env, chef_id, position, station)
        result = act(env, chef_id, action)
        if action == INTERACT:
            return result
    raise AssertionError("Station could not be reached")


def prepared(env):
    controller = ScriptedController(env)
    interact_at(env, controller.main, (4, 1), STATIONS["M"])
    interact_at(env, controller.main, (4, 2), STATIONS["G"])
    while env.state.cooker.remaining:
        env.step([STAY] * 3)
    interact_at(env, controller.side, (5, 1), STATIONS["L"])
    interact_at(env, controller.side, (5, 2), STATIONS["K"])
    for _ in range(100):
        action = navigation_action(env, controller.server, (2, 2), STATIONS["G"])
        if action == INTERACT:
            return controller
        act(env, controller.server, action)
    raise AssertionError("Unloading position could not be reached")


def rollout(config=None, seed=0):
    env = SharedKitchen(config)
    env.reset(seed=seed)
    controller = ScriptedController(env)
    trace = []
    while not (env.terminated or env.truncated):
        actions = controller.actions(env)
        result = env.step(actions)
        trace.append((actions, result[4]))
    return env, controller, trace


class SharedKitchenTests(unittest.TestCase):
    def joint(self, **kwargs):
        return SharedKitchen(SharedKitchenConfig(normal_orders=0, joint_orders=1, shuffle_players=False, **kwargs))

    def normal(self, **kwargs):
        return SharedKitchen(SharedKitchenConfig(normal_orders=1, joint_orders=0, shuffle_players=False, **kwargs))

    def test_exact_layout_and_connectivity(self):
        self.assertEqual(LAYOUT, ("XXXXMXX", "X     L", "X  G  K", "S  A  X", "X  B  X", "D     X", "XXXXXXX"))
        self.assertEqual(len(FLOOR), 22)
        for occupied in (set(), set(COOKER_PORTS)):
            available = FLOOR - occupied
            seen = {next(iter(available))}
            while True:
                extended = seen | {q for p in seen for q in available if abs(p[0] - q[0]) + abs(p[1] - q[1]) == 1}
                if seen == extended:
                    break
                seen = extended
            self.assertEqual(seen, available)

    def test_full_mixed_episode_from_empty(self):
        env, controller, trace = rollout(SharedKitchenConfig(normal_orders=4, joint_orders=4))
        self.assertTrue(env.terminated)
        self.assertFalse(env.truncated)
        self.assertEqual(env.state.score, 160)
        self.assertEqual(env.state.metrics["normal_completed"], 4)
        self.assertEqual(env.state.metrics["joint_completed"], 4)
        self.assertEqual(sum(info["team_reward"] for _, info in trace), 160)
        self.assertEqual(len({tuple(order.delivered_units) for order in env.state.orders}), 8)
        for order in env.state.orders:
            self.assertEqual(order.contributors, (0, 1, 2))
            self.assertEqual(order.side_prepared_by, (controller.side,))
            self.assertEqual(len(order.unloaded_by), 1 if order.kind == "normal" else 2)
        self.assertTrue(any(env.state.orders[i + 1].started_at < env.state.orders[i].completed_at for i in range(7)))

    def test_continuous_orders_continue_past_eight_and_stop_at_horizon(self):
        env = SharedKitchen(SharedKitchenConfig(shuffle_players=False))
        fixed = SharedKitchen(SharedKitchenConfig(normal_orders=4, joint_orders=4, shuffle_players=False))
        controller = ScriptedController(env)
        initial_shape = fixed.centralized_state().shape
        self.assertEqual(initial_shape, (863,))
        np.testing.assert_array_equal(env.observations(), fixed.observations())
        stepped_past_eight = False
        while not (env.terminated or env.truncated):
            actions = controller.actions(env)
            _, _, terminated, truncated, info = env.step(actions)
            self.assertFalse(terminated)
            self.assertEqual(env.centralized_state().shape, initial_shape)
            if len(env.state.orders) <= 8:
                fixed.step(actions)
                np.testing.assert_array_equal(env.observations(), fixed.observations())
            if info["metrics"]["meals_completed"] == 8 and not stepped_past_eight:
                self.assertFalse(truncated)
                outstanding = [order for order in env.state.orders if order.status != "served"]
                self.assertEqual(len(outstanding), env.config.active_limit)
                restored = SharedKitchen(env.config)
                restored.restore(json.loads(json.dumps(env.snapshot())))
                np.testing.assert_array_equal(restored.centralized_state(), env.centralized_state())
                stepped_past_eight = True
        self.assertTrue(stepped_past_eight)
        self.assertTrue(env.truncated)
        self.assertEqual(env.state.tick, 300)
        self.assertGreater(info["metrics"]["meals_completed"], 8)
        self.assertEqual(info["metrics"]["team_score"], 20 * info["metrics"]["meals_completed"])
        self.assertEqual(info["metrics"]["joint_wait_steps"],
                         info["metrics"]["joint_ready_steps"] - info["metrics"]["joint_successes"])
        self.assertGreater(info["metrics"]["joint_wait_steps"], 0)
        self.assertNotIn("completion_rate", info["metrics"])
        self.assertEqual([order.kind for order in env.state.orders[:10]], ["normal", "joint"] * 5)

        first_delivery_tick = env.state.orders[0].completed_at
        exact, _, _ = rollout(SharedKitchenConfig(horizon=first_delivery_tick, shuffle_players=False))
        self.assertTrue(exact.truncated)
        self.assertFalse(exact.terminated)
        self.assertEqual(exact.state.score, 20)
        self.assertEqual(len(exact.state.orders), exact.config.active_limit)
        self.assertEqual(exact.centralized_state().shape, initial_shape)

    def test_observation_reuses_served_slots_without_moving_a_started_order(self):
        env = SharedKitchen()
        env.state.orders = [Ticket(0, "normal", status="started", released_at=0)]
        for order_id in range(1, 10):
            kind = ("normal", "joint")[order_id % 2]
            env.state.orders.append(Ticket(order_id, kind, status="served",
                                           released_at=order_id - 1, completed_at=order_id))
        env.state.orders.append(Ticket(10, "normal", status="active", released_at=9))
        slots = env._observed_orders(env.state)
        self.assertEqual([order.id for order in slots], [0, 8, 9, 10, 4, 5, 6, 7])

    def test_normal_only_and_joint_only(self):
        for normal, joint in ((3, 0), (0, 3)):
            with self.subTest(normal=normal, joint=joint):
                env, _, _ = rollout(SharedKitchenConfig(normal_orders=normal, joint_orders=joint))
                self.assertTrue(env.terminated)
                self.assertEqual(env.state.score, 60)

    def test_every_player_permutation(self):
        seen = set()
        for seed in range(50):
            env = SharedKitchen(SharedKitchenConfig(normal_orders=1, joint_orders=1))
            env.reset(seed)
            positions = tuple(chef.position for chef in env.state.chefs)
            if positions in seen:
                continue
            seen.add(positions)
            controller = ScriptedController(env)
            while not (env.terminated or env.truncated):
                env.step(controller.actions(env))
            self.assertTrue(env.terminated, positions)
            self.assertTrue(all(order.contributors == (0, 1, 2) for order in env.state.orders))
            if len(seen) == 6:
                break
        self.assertEqual(len(seen), len(tuple(itertools.permutations(range(3)))))

    def test_solo_chef_can_complete_normal_meal(self):
        env = self.normal(active_players=(0,))
        interact_at(env, 0, (4, 1), STATIONS["M"])
        interact_at(env, 0, (4, 2), STATIONS["G"])
        while env.state.cooker.remaining:
            env.step([STAY] * 3)
        interact_at(env, 0, (4, 2), STATIONS["G"])
        interact_at(env, 0, (5, 1), STATIONS["L"])
        for _ in range(env.config.chop_steps + 2):
            interact_at(env, 0, (5, 2), STATIONS["K"])
        interact_at(env, 0, (4, 3), STATIONS["A"])
        interact_at(env, 0, (1, 5), STATIONS["D"])
        interact_at(env, 0, (2, 3), STATIONS["A"])
        interact_at(env, 0, (2, 3), STATIONS["A"])
        self.assertEqual(env.state.assembly.plate.kind, "meal")
        self.assertIsNone(env.state.chefs[0].held)
        interact_at(env, 0, (2, 3), STATIONS["A"])
        interact_at(env, 0, (1, 3), STATIONS["S"])
        self.assertTrue(env.terminated)
        self.assertEqual(env.order(0).contributors, (0,))

    def test_joint_unload_and_third_chef_chop_same_step_same_meal(self):
        env = self.joint()
        controller = prepared(env)
        side_unit = env.state.cutting.item.units[0]
        actions = [INTERACT] * 3
        info = env.step(actions)[4]
        self.assertEqual({event["type"] for event in info["events"]}, {"chop", "unload"})
        self.assertEqual(env.state.cutting.progress, 1)
        self.assertIsNone(env.state.cooker.item)
        self.assertEqual(env.order(0).unloaded_by, tuple(sorted((controller.main, controller.server))))
        while not (env.terminated or env.truncated):
            env.step(controller.actions(env))
        order = env.order(0)
        self.assertTrue(env.terminated)
        self.assertEqual(order.delivered_units[1], side_unit)
        self.assertEqual(order.side_prepared_by, (controller.side,))
        self.assertNotIn(controller.side, order.unloaded_by)
        self.assertEqual(order.contributors, (0, 1, 2))

    def test_alternating_actions_do_not_accumulate_joint_progress(self):
        env = self.joint()
        controller = prepared(env)
        for i in (controller.server, controller.main) * 4:
            act(env, i, INTERACT)
            self.assertIsNotNone(env.state.cooker.item)
            self.assertIsNone(env.state.assembly.main)
        actions = [STAY] * 3
        actions[controller.server] = actions[controller.main] = INTERACT
        env.step(actions)
        self.assertIsNone(env.state.cooker.item)
        self.assertEqual(env.state.metrics["joint_successes"], 1)
        env.step(actions)
        self.assertEqual(env.state.metrics["joint_successes"], 1)

    def test_joint_rejects_full_hands_and_wrong_facing(self):
        env = self.joint()
        controller = prepared(env)
        base = env.snapshot()
        env.state.chefs[controller.server].orientation = (0, -1)
        env.step([INTERACT] * 3)
        self.assertIsNotNone(env.state.cooker.item)
        env.restore(base)
        interact_at(env, controller.server, (1, 5), STATIONS["D"])
        interact_at(env, controller.server, (2, 2), STATIONS["G"])
        env.step([INTERACT] * 3)
        self.assertIsNotNone(env.state.cooker.item)
        self.assertIsNone(env.state.assembly.main)

    def test_north_floor_is_walkable_but_cannot_load(self):
        env = self.normal(active_players=(0,))
        interact_at(env, 0, (4, 1), STATIONS["M"])
        interact_at(env, 0, (3, 1), STATIONS["G"])
        self.assertEqual(env.state.chefs[0].position, (3, 1))
        self.assertIsNone(env.state.cooker.item)
        self.assertEqual(env.state.chefs[0].held.kind, "raw_main")

    def test_north_floor_cannot_replace_either_joint_position(self):
        env = self.joint()
        controller = prepared(env)
        interact_at(env, controller.side, (3, 1), STATIONS["G"])
        actions = [STAY] * 3
        actions[controller.side] = actions[controller.server] = INTERACT
        env.step(actions)
        self.assertIsNotNone(env.state.cooker.item)
        actions[controller.main] = INTERACT
        env.step(actions)
        self.assertIsNone(env.state.cooker.item)
        self.assertNotIn(controller.side, env.order(0).unloaded_by)

    def test_north_floor_cannot_unload_normal_meal(self):
        env = self.normal(active_players=(0,))
        interact_at(env, 0, (4, 1), STATIONS["M"])
        interact_at(env, 0, (4, 2), STATIONS["G"])
        while env.state.cooker.remaining:
            env.step([STAY] * 3)
        interact_at(env, 0, (3, 1), STATIONS["G"])
        self.assertIsNotNone(env.state.cooker.item)
        self.assertIsNone(env.state.assembly.main)

    def test_new_load_and_new_side_wait_for_next_step(self):
        env = self.normal()
        controller = ScriptedController(env)
        interact_at(env, controller.main, (4, 1), STATIONS["M"])
        interact_at(env, controller.main, (4, 2), STATIONS["G"])
        self.assertEqual(env.state.cooker.remaining, env.config.cook_steps)
        env.step([STAY] * 3)
        self.assertEqual(env.state.cooker.remaining, env.config.cook_steps - 1)
        interact_at(env, controller.side, (5, 1), STATIONS["L"])
        interact_at(env, controller.side, (5, 2), STATIONS["K"])
        self.assertEqual(env.state.cutting.progress, 0)
        self.assertEqual(env.state.cutting.item.kind, "raw_side")
        act(env, controller.side, INTERACT)
        self.assertEqual(env.state.cutting.progress, 1)
        self.assertIsNone(env.state.chefs[controller.side].held)

    def test_separate_assembly_slots_can_be_deposited_concurrently(self):
        env = self.joint()
        controller = prepared(env)
        actions = [STAY] * 3
        actions[controller.main] = actions[controller.server] = INTERACT
        env.step(actions)
        for _ in range(env.config.chop_steps + 1):
            act(env, controller.side, INTERACT)
        interact_at(env, controller.server, (1, 5), STATIONS["D"])
        for i, position in ((controller.server, (2, 3)), (controller.side, (4, 3))):
            for _ in range(100):
                action = navigation_action(env, i, position, STATIONS["A"])
                if action == INTERACT:
                    break
                act(env, i, action)
        actions = [STAY] * 3
        actions[controller.server] = actions[controller.side] = INTERACT
        info = env.step(actions)[4]
        self.assertEqual([event["type"] for event in info["events"]], ["place_assembly", "place_assembly"])
        self.assertEqual(env.state.assembly.side.kind, "chopped_side")
        self.assertEqual(env.state.assembly.plate.kind, "plate")
        env.step(actions)
        self.assertEqual(env.state.assembly.plate.kind, "plate")
        act(env, controller.server, INTERACT)
        self.assertEqual(env.state.assembly.plate.kind, "meal")
        self.assertIsNone(env.state.chefs[controller.server].held)
        act(env, controller.server, INTERACT)
        self.assertEqual(env.state.chefs[controller.server].held.kind, "meal")
        self.assertIsNone(env.state.assembly.plate)

    def test_same_slot_conflicts_preserve_both_plates(self):
        env = self.normal()
        controller = ScriptedController(env)
        for i, position in ((controller.server, (2, 3)), (controller.main, (4, 3))):
            interact_at(env, i, (1, 5), STATIONS["D"])
            for _ in range(100):
                action = navigation_action(env, i, position, STATIONS["A"])
                if action == INTERACT:
                    break
                act(env, i, action)
        actions = [STAY] * 3
        actions[controller.main] = actions[controller.server] = INTERACT
        env.step(actions)
        self.assertIsNone(env.state.assembly.plate)
        self.assertEqual(env.state.chefs[controller.main].held.kind, "plate")
        self.assertEqual(env.state.chefs[controller.server].held.kind, "plate")
        act(env, controller.server, INTERACT)
        self.assertEqual(env.state.assembly.plate.kind, "plate")

    def test_buffer_does_not_relay_new_deposit_in_same_step(self):
        env = self.normal()
        controller = ScriptedController(env)
        interact_at(env, controller.side, (5, 1), STATIONS["L"])
        for i, goal in ((controller.side, (4, 4)), (controller.server, (2, 4))):
            for _ in range(100):
                action = navigation_action(env, i, goal, STATIONS["B"])
                if action == INTERACT:
                    break
                act(env, i, action)
        actions = [STAY] * 3
        actions[controller.side] = actions[controller.server] = INTERACT
        env.step(actions)
        self.assertIsNotNone(env.state.buffer)
        self.assertIsNone(env.state.chefs[controller.server].held)
        self.assertIsNone(env.state.chefs[controller.side].held)
        env.step(actions)
        self.assertIsNotNone(env.state.buffer)
        act(env, controller.server, INTERACT)
        self.assertIsNone(env.state.buffer)
        self.assertEqual(env.state.chefs[controller.server].held.kind, "raw_side")

    def test_blocked_assembly_main_does_not_destroy_next_main(self):
        env = SharedKitchen(SharedKitchenConfig(normal_orders=2, joint_orders=0, shuffle_players=False))
        controller = prepared(env)
        act(env, controller.main, INTERACT)
        first = env.state.assembly.main
        interact_at(env, controller.main, (4, 1), STATIONS["M"])
        interact_at(env, controller.main, (4, 2), STATIONS["G"])
        while env.state.cooker.remaining:
            env.step([STAY] * 3)
        second = env.state.cooker.item
        env.step([INTERACT] * 3)
        self.assertEqual(env.state.assembly.main, first)
        self.assertEqual(env.state.cooker.item, second)

    def test_simultaneous_movement_chain_and_swap(self):
        env = self.normal()
        for chef, position in zip(env.state.chefs, ((1, 1), (2, 1), (3, 1))):
            chef.position = position
        env.step([EAST, WEST, STAY])
        self.assertEqual([chef.position for chef in env.state.chefs], [(1, 1), (2, 1), (3, 1)])
        env.step([EAST, EAST, EAST])
        self.assertEqual([chef.position for chef in env.state.chefs], [(2, 1), (3, 1), (4, 1)])

    def test_same_destination_and_blocked_chain(self):
        env = self.normal()
        for chef, position in zip(env.state.chefs, ((2, 1), (4, 1), (1, 5))):
            chef.position = position
        env.step([EAST, WEST, STAY])
        self.assertEqual([chef.position for chef in env.state.chefs], [(2, 1), (4, 1), (1, 5)])
        for chef, position in zip(env.state.chefs, ((1, 1), (2, 1), (3, 1))):
            chef.position = position
        env.step([EAST, EAST, STAY])
        self.assertEqual([chef.position for chef in env.state.chefs], [(1, 1), (2, 1), (3, 1)])

    def test_material_returns_and_surplus_side_recycling(self):
        for source, station, position in (("raw_main", "M", (4, 1)), ("raw_side", "L", (5, 1)), ("plate", "D", (1, 5))):
            env = self.normal(active_players=(0,))
            interact_at(env, 0, position, STATIONS[station])
            self.assertEqual(env.state.chefs[0].held.kind, source)
            act(env, 0, INTERACT)
            self.assertIsNone(env.state.chefs[0].held)
            self.assertEqual(env.state.returned_units, [0])
            self.assertEqual(env.state.score, 0)
        env = self.normal(active_players=(0,))
        interact_at(env, 0, (5, 1), STATIONS["L"])
        for _ in range(env.config.chop_steps + 2):
            interact_at(env, 0, (5, 2), STATIONS["K"])
        interact_at(env, 0, (5, 1), STATIONS["L"])
        self.assertEqual(env.state.wasted_units, [0])
        self.assertEqual(env.state.score, 0)

    def test_snapshot_json_replay_and_rng(self):
        env = self.joint()
        controller = prepared(env)
        snapshot = json.loads(json.dumps(env.snapshot()))
        actions = [INTERACT] * 3
        expected = env.step(actions)
        expected_state = env.snapshot()["state"]
        expected_reset = env.reset()[0]
        env.restore(snapshot)
        actual = env.step(actions)
        self.assertEqual(env.snapshot()["state"], expected_state)
        np.testing.assert_array_equal(actual[0], expected[0])
        self.assertEqual(actual[1:], expected[1:])
        actual_reset = env.reset()[0]
        np.testing.assert_array_equal(actual_reset, expected_reset)
        self.assertNotEqual(controller.side, controller.main)

    def test_invalid_restore_is_atomic(self):
        env = self.normal()
        controller = ScriptedController(env)
        interact_at(env, controller.main, (4, 1), STATIONS["M"])
        before = env.snapshot()
        malformed = copy.deepcopy(before)
        malformed["state"]["buffer"] = copy.deepcopy(malformed["state"]["chefs"][controller.main]["held"])
        with self.assertRaises(ValueError):
            env.restore(malformed)
        self.assertEqual(env.snapshot(), before)
        malformed = copy.deepcopy(before)
        malformed["fingerprint"] = "wrong"
        with self.assertRaises(ValueError):
            env.restore(malformed)
        self.assertEqual(env.snapshot(), before)

    def test_invalid_actions_are_atomic(self):
        env = self.normal()
        before = env.snapshot()
        for actions in ((True, STAY, STAY), (np.bool_(False), STAY, STAY), (6, STAY, STAY), (1.0, STAY, STAY), (STAY,), None):
            with self.subTest(actions=actions):
                with self.assertRaises(ValueError):
                    env.step(actions)
                self.assertEqual(env.snapshot(), before)

    def test_random_actions_preserve_materials_and_observations(self):
        for seed in range(10):
            env = SharedKitchen()
            env.reset(seed)
            rng = np.random.default_rng(seed)
            while not (env.terminated or env.truncated):
                observations = env.step(rng.integers(0, 6, size=3))[0]
                self.assertTrue(np.isfinite(observations).all())
                self.assertTrue((np.asarray(observations) >= 0).all())
                self.assertTrue((np.asarray(observations) <= 1).all())
            self.assertTrue(env.validate_state())

    def test_completion_on_last_step_beats_truncation(self):
        config = SharedKitchenConfig(normal_orders=1, joint_orders=1, shuffle_players=False)
        first, _, _ = rollout(config)
        count = first.state.tick
        exact, _, _ = rollout(SharedKitchenConfig(normal_orders=1, joint_orders=1, shuffle_players=False, horizon=count))
        self.assertTrue(exact.terminated)
        self.assertFalse(exact.truncated)
        early, _, _ = rollout(SharedKitchenConfig(normal_orders=1, joint_orders=1, shuffle_players=False, horizon=count - 1))
        self.assertFalse(early.terminated)
        self.assertTrue(early.truncated)
        self.assertEqual(early.state.score, 20)
        with self.assertRaises(RuntimeError):
            early.step([STAY] * 3)

    def test_config_validation(self):
        for data in ({"cook_steps": 0}, {"horizon": True}, {"normal_orders": -1}, {"normal_orders": 0, "joint_orders": 0}, {"joint_reward": float("nan")}, {"active_players": (0, 0)}, {"active_players": (True,)}, {"shuffle_players": 1}):
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    SharedKitchenConfig(**data)

    def test_parallel_api_seeding_and_shared_reward(self):
        from pettingzoo.test import parallel_api_test, parallel_seed_test
        from overcooked_ai_py.cooperative.shared.parallel import ParallelSharedKitchen
        config = SharedKitchenConfig(normal_orders=1, joint_orders=1, horizon=120)
        parallel_api_test(ParallelSharedKitchen(config), num_cycles=130)
        parallel_seed_test(lambda: ParallelSharedKitchen(config), num_cycles=130)
        env = ParallelSharedKitchen(config)
        env.reset(seed=1)
        controller = ScriptedController(env.core)
        team_reward = 0
        while env.agents:
            joint = controller.actions(env.core)
            _, rewards, _, _, infos = env.step({agent: joint[int(agent[-1])] for agent in env.agents})
            self.assertEqual(len(set(rewards.values())), 1)
            self.assertTrue(all(info["team_reward"] == next(iter(rewards.values())) for info in infos.values()))
            team_reward += next(iter(rewards.values()))
            self.assertTrue(env.state_space.contains(env.state()))
        self.assertEqual(team_reward, 40)
        self.assertEqual(env.step({}), ({}, {}, {}, {}, {}))

    def test_new_files_have_no_comments_or_docstrings(self):
        import ast
        import overcooked_ai_py.cooperative.shared as package
        paths = list(Path(package.__file__).parent.glob("*.py")) + [Path(__file__), Path(__file__).resolve().parents[1] / "scripts" / "demo_shared_kitchen.py"]
        for path in paths:
            content = path.read_text(encoding="utf-8")
            self.assertFalse(any(token.type == tokenize.COMMENT for token in tokenize.generate_tokens(io.StringIO(content).readline)), str(path))
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.assertIsNone(ast.get_docstring(node), str(path))


if __name__ == "__main__":
    unittest.main()
