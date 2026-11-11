"""Behavioral, conservation, replay, and task-design checks for MixedKitchen."""

import copy
import importlib.util
import json
import unittest

import numpy as np

from overcooked_ai_py.cooperative import KitchenConfig, MixedKitchen
from overcooked_ai_py.cooperative.controller import ReferenceController
from overcooked_ai_py.cooperative.diagnostics import overlap_probe, single_category_score_bounds
from overcooked_ai_py.cooperative.kitchen import INTERACT as I, STAY as W
from overcooked_ai_py.cooperative.model import Item, Pot
from overcooked_ai_py.cooperative.presets import preset
from overcooked_ai_py.cooperative.parallel import ParallelKitchen


def ready_handoff(env):
    state = env.state
    state.chefs[0].position, state.chefs[0].orientation = (4, 3), (1, 0)
    state.chefs[1].position, state.chefs[1].orientation = (6, 3), (-1, 0)
    state.chefs[2].position = (9, 3)
    state.orders[1].stock = 0
    state.chefs[0].held = Item("soup", state.orders[1].id)
    env.validate_state()


class TestMixedKitchen(unittest.TestCase):
    def make(self, **kwargs):
        return MixedKitchen(KitchenConfig(shuffle_players=False, deadline_jitter=0,
                                         duration_jitter=0, **kwargs))

    def test_configuration_rejects_invalid_rules(self):
        for params in ({"sync_steps": 0}, {"duration_jitter": 4},
                       {"normal_reward": float("nan")}, {"joint_reward": -1},
                       {"active_players": (0, 0)}, {"active_players": (3,)},
                       {"sync_deadline": 100}, {"layout": "unknown"}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                KitchenConfig(**params)

    def test_fixed_observation_and_all_active_agents_visible(self):
        env = self.make()
        obs, _ = env.reset(seed=0)
        self.assertEqual([x.shape for x in obs], [(187,)] * 3)
        for x in obs:
            self.assertEqual(x.dtype, np.float32)
            self.assertTrue(np.isfinite(x).all())
            self.assertTrue(((x >= 0) & (x <= 1)).all())
        self.assertTrue(np.array_equal(env.centralized_state(), obs[0]))
        self.assertFalse(np.array_equal(obs[0], obs[1]))

    def test_parallel_dictionary_interface_and_shared_reward(self):
        wrapped = ParallelKitchen(KitchenConfig(horizon=2))
        obs, infos = wrapped.reset(seed=0, options={"options": 1})
        for agent in wrapped.agents:
            self.assertTrue(wrapped.observation_space(agent).contains(obs[agent]))
        self.assertTrue(wrapped.state_space.contains(wrapped.state()))
        with self.assertRaises(ValueError):
            wrapped.step({"chef_0": W})
        for _ in range(2):
            obs, rewards, ends, cuts, infos = wrapped.step({a: W for a in wrapped.agents})
            self.assertEqual(len(set(rewards.values())), 1)
        self.assertTrue(all(ends.values()))
        self.assertFalse(any(cuts.values()))
        self.assertEqual(wrapped.agents, [])
        self.assertEqual(wrapped.step({}), ({}, {}, {}, {}, {}))

    @unittest.skipUnless(importlib.util.find_spec("pettingzoo"), "optional PettingZoo not installed")
    def test_official_parallel_api_for_three_and_two_chefs(self):
        from pettingzoo.test import parallel_api_test
        for active in ((0, 1, 2), (0, 2)):
            parallel_api_test(ParallelKitchen(KitchenConfig(active_players=active)), num_cycles=400)

    def test_unfinished_orders_are_included_in_closing_statistics(self):
        env = self.make(horizon=1)
        _, _, done, _, info = env.step((W, W, W))
        self.assertTrue(done)
        self.assertEqual(info["metrics"]["normal_issued"], 1)
        self.assertEqual(info["metrics"]["joint_issued"], 1)
        self.assertEqual(info["metrics"]["unfinished_at_close"], 2)

    def test_reset_seed_and_snapshot_exact_replay(self):
        a, b = MixedKitchen(), MixedKitchen()
        a.reset(seed=19)
        b.reset(seed=19)
        actions = np.random.default_rng(4).integers(0, 6, (150, 3))
        for index, joint_action in enumerate(actions):
            if index == 75:
                # JSON serialization includes generator state, not just positions.
                saved = json.loads(json.dumps(a.snapshot()))
                b.reset(seed=99)
                b.restore(saved)
            ra, rb = a.step(joint_action), b.step(joint_action)
            self.assertEqual(ra[1:], rb[1:])
            self.assertEqual(a.snapshot(), b.snapshot())
            for oa, ob in zip(ra[0], rb[0]):
                np.testing.assert_array_equal(oa, ob)

    def test_interleaved_instances_do_not_share_task_or_random_state(self):
        a, reference = MixedKitchen(), MixedKitchen()
        different = MixedKitchen(KitchenConfig(cook_steps=7, sync_steps=6, layout="offset"))
        a.reset(seed=15)
        reference.reset(seed=15)
        rng = np.random.default_rng(20)
        for tick in range(150):
            if tick % 20 == 0:
                different.reset(seed=tick)
            different.step(rng.integers(0, 6, 3))
            actions = rng.integers(0, 6, 3)
            a.step(actions)
            reference.step(actions)
            self.assertEqual(a.snapshot(), reference.snapshot())

    def test_invalid_actions_do_not_mutate_state_or_rng(self):
        env = self.make()
        before = env.snapshot()
        for actions in ((W, W), (W, W, W, W), (6, W, W), (True, W, W), (1.0, W, W)):
            with self.subTest(actions=actions), self.assertRaises(ValueError):
                env.step(actions)
            self.assertEqual(env.snapshot(), before)

    def test_continuous_operation_and_no_free_sync_reward(self):
        env = self.make()
        ready_handoff(env)
        for tick in range(3):
            _, reward, _, _, _ = env.step((I, I, W))
            self.assertEqual(reward, 0)
            self.assertEqual(env.state.joint.progress, tick + 1)
            self.assertFalse(env.state.chefs[0].held.certified)
        env.step((I, I, W))
        self.assertIsNone(env.state.chefs[0].held)
        self.assertTrue(env.state.chefs[1].held.certified)
        self.assertEqual(env.state.metrics["joint_completions"], 1)

    def test_interruption_loses_progress_and_cannot_be_staggered(self):
        env = self.make()
        ready_handoff(env)
        env.step((I, I, W))
        env.step((I, W, W))
        self.assertEqual(env.state.joint.progress, 0)
        self.assertEqual(env.state.metrics["joint_aborts"], 1)
        env.step((W, I, W))
        self.assertEqual(env.state.joint.progress, 0)
        env.step((I, I, W))
        self.assertEqual(env.state.joint.progress, 1)

    def test_any_pair_can_operate_independent_of_player_number(self):
        for mapping in ((0, 1, 2), (2, 0, 1), (1, 2, 0)):
            env = self.make()
            ready_handoff(env)
            old = copy.deepcopy(env.state.chefs)
            for new_index, old_index in enumerate(mapping):
                env.state.chefs[new_index] = old[old_index]
            donor = mapping.index(0)
            receiver = mapping.index(1)
            actions = [W, W, W]
            actions[donor] = actions[receiver] = I
            for _ in range(4):
                env.step(actions)
            self.assertTrue(env.state.chefs[receiver].held.certified)
            self.assertIsNone(env.state.chefs[donor].held)

    def test_third_chef_can_work_during_joint_operation(self):
        env = self.make()
        ready_handoff(env)
        env.state.chefs[2].position, env.state.chefs[2].orientation = (9, 1), (1, 0)
        env.step((I, I, I))
        self.assertEqual(env.state.joint.progress, 1)
        self.assertEqual(env.state.chefs[2].held.kind, "ingredient")
        self.assertEqual(env.state.chefs[2].held.order_id, env.state.orders[0].id)

    def test_invalid_items_cannot_be_certified(self):
        for kind, slot in (("ingredient", 1), ("soup", 0), ("dish", None)):
            env = self.make()
            ready_handoff(env)
            # Return the previous joint soup to stock before replacing the held item.
            env.state.orders[1].stock = 1
            if slot is not None:
                env.state.orders[slot].stock = 0
            env.state.chefs[0].held = Item(kind, env.state.orders[slot].id if slot is not None else None)
            env.step((I, I, W))
            self.assertEqual(env.state.joint.progress, 0)

    def test_full_receiver_cannot_accept(self):
        env = self.make()
        ready_handoff(env)
        env.state.chefs[1].held = Item("dish")
        env.step((I, I, W))
        self.assertEqual(env.state.joint.progress, 0)
        self.assertEqual(env.state.chefs[0].held.kind, "soup")
        self.assertEqual(env.state.chefs[1].held.kind, "dish")

    def test_uncertified_joint_soup_cannot_bypass_handoff(self):
        env = self.make()
        ready_handoff(env)
        env.state.chefs[0].position, env.state.chefs[0].orientation = (9, 5), (1, 0)
        for _ in range(3):
            _, reward, _, _, info = env.step((I, W, W))
            self.assertEqual(reward, 0)
            self.assertTrue(any(e["type"] == "delivery_rejected" for e in info["events"]))
        self.assertIsNotNone(env.state.chefs[0].held)

    def test_certified_soup_cannot_farm_handoff_rewards(self):
        env = self.make()
        ready_handoff(env)
        for _ in range(12):
            env.step((I, I, W))
        self.assertEqual(env.state.metrics["joint_completions"], 1)
        self.assertEqual(env.state.score, 0)
        self.assertTrue(env.state.chefs[1].held.certified)

    def test_ingredient_stock_is_conserved_on_repeated_pickup(self):
        env = self.make()
        chef = env.state.chefs[0]
        chef.position, chef.orientation = (9, 1), (1, 0)
        env.step((I, W, W))
        chef = env.state.chefs[0]
        chef.position, chef.orientation = (4, 2), (1, 0)
        env.step((I, W, W))
        chef = env.state.chefs[0]
        chef.position, chef.orientation = (9, 1), (1, 0)
        env.step((I, W, W))
        self.assertIsNone(env.state.chefs[0].held)
        self.assertIsNotNone(env.state.buffers[0])
        self.assertEqual(env.state.orders[0].stock, 0)

    def test_full_hands_and_full_buffers_have_a_recovery_action(self):
        env = self.make()
        for chef in env.state.chefs:
            chef.held = Item("dish")
        env.state.buffers = [Item("dish"), Item("dish")]
        env.state.chefs[0].position, env.state.chefs[0].orientation = (1, 5), (-1, 0)
        _, reward, _, _, _ = env.step((I, W, W))
        self.assertIsNone(env.state.chefs[0].held)
        self.assertEqual(reward, 0)
        env.state.chefs[0].position, env.state.chefs[0].orientation = (1, 1), (-1, 0)
        env.step((I, W, W))
        self.assertEqual(env.state.chefs[0].held.kind, "ingredient")

    def test_returned_ingredient_does_not_duplicate_stock(self):
        env = self.make()
        env.state.chefs[0].position, env.state.chefs[0].orientation = (1, 1), (-1, 0)
        for _ in range(8):
            env.step((I, W, W))
            env.validate_state()
        self.assertIsNone(env.state.chefs[0].held)
        self.assertEqual(env.state.orders[1].stock, 1)
        self.assertEqual(env.state.score, 0)

    def test_prepared_curricula_preserve_rules_and_have_distinct_fingerprints(self):
        standard = MixedKitchen(preset("standard_eval"))
        fingerprints = {standard.fingerprint}
        for name in ("handoff_1", "handoff_2", "handoff_4", "kitchen_ready", "full_easy"):
            env = MixedKitchen(preset(name))
            env.validate_state()
            self.assertEqual(env.observations()[0].shape, (187,))
            fingerprints.add(env.fingerprint)
        self.assertEqual(len(fingerprints), 6)

    def test_third_player_capacity_probe_and_single_category_bounds(self):
        self.assertEqual(overlap_probe()["score"], 60)
        self.assertEqual(overlap_probe((0, 1))["score"], 25)
        bounds = single_category_score_bounds(KitchenConfig())
        self.assertEqual(bounds["normal_only"], 80)
        self.assertEqual(bounds["joint_only"], 120)

    def test_cooking_requires_elapsed_steps(self):
        env = self.make()
        env.state.orders[0].stock = 0
        chef = env.state.chefs[0]
        chef.held = Item("ingredient", env.state.orders[0].id)
        chef.position, chef.orientation = (8, 1), (0, -1)
        env.step((I, W, W))
        self.assertEqual(env.state.pots[1].remaining, 4)
        env.state.chefs[0].held = Item("dish")
        for _ in range(4):
            env.step((I, W, W))
            self.assertEqual(env.state.chefs[0].held.kind, "dish")
        env.step((I, W, W))
        self.assertEqual(env.state.chefs[0].held.kind, "soup")

    def test_deadlines_include_exact_boundary_but_not_next_step(self):
        env = self.make()
        env.state.orders[0].stock = 0
        env.state.orders[0].due_at = 1
        chef = env.state.chefs[0]
        chef.held = Item("soup", env.state.orders[0].id)
        chef.position, chef.orientation = (9, 5), (1, 0)
        env.step((I, W, W))
        self.assertEqual(env.state.score, 20)
        env = self.make()
        env.state.orders[0].due_at = 1
        env.step((W, W, W))
        env.state.orders[0].stock = 0
        chef = env.state.chefs[0]
        chef.held = Item("soup", env.state.orders[0].id)
        chef.position, chef.orientation = (9, 5), (1, 0)
        env.step((I, W, W))
        self.assertEqual(env.state.orders[0].status, "expired")
        self.assertEqual(env.state.score, -5)

    def test_late_joint_completion_cannot_extend_its_deadline(self):
        env = self.make()
        ready_handoff(env)
        env.state.orders[1].sync_due_at = 3
        for _ in range(4):
            env.step((I, I, W))
        self.assertEqual(env.state.metrics["joint_completions"], 0)
        self.assertEqual(env.state.orders[1].status, "expired")
        self.assertIsNone(env.state.chefs[0].held)
        self.assertEqual(env.state.joint.progress, 0)

    def test_completed_normal_order_does_not_refresh_on_its_own(self):
        env = self.make()
        env.state.orders[0].stock = 0
        chef = env.state.chefs[0]
        chef.held = Item("soup", env.state.orders[0].id)
        chef.position, chef.orientation = (9, 5), (1, 0)
        env.step((I, W, W))
        for _ in range(20):
            env.step((W, W, W))
        self.assertEqual(env.state.batch, 1)
        self.assertEqual(env.state.orders[0].status, "completed")
        self.assertEqual(env.state.metrics["normal_completed"], 1)

    def test_batch_bonus_is_paid_once_and_end_cannot_be_stepped(self):
        env = self.make(max_batches=1)
        policy = ReferenceController()
        total = 0
        for _ in range(100):
            _, reward, done, _, _ = env.step(policy(env))
            total += reward
            if done:
                break
        self.assertTrue(done)
        self.assertEqual(total, 60)
        before = env.snapshot()
        with self.assertRaises(RuntimeError):
            env.step((I, I, I))
        self.assertEqual(before, env.snapshot())

    def test_potential_shaping_telescopes_and_returns_raw_score_separately(self):
        env = self.make(horizon=6, shaping_scale=0.5)
        ready_handoff(env)
        initial = env.potential()
        discounted = 0
        for t in range(6):
            _, reward, done, _, info = env.step((I, I, W))
            self.assertEqual(info["raw_reward"], 0)
            discounted += env.config.shaping_gamma ** t * info["shaping_reward"]
        self.assertTrue(done)
        self.assertAlmostEqual(discounted, -0.5 * initial, places=6)

    def test_contested_buffer_is_not_resolved_by_player_number(self):
        env = self.make()
        env.state.buffers[0] = Item("dish")
        env.state.chefs[0].position, env.state.chefs[0].orientation = (4, 2), (1, 0)
        env.state.chefs[1].position, env.state.chefs[1].orientation = (6, 2), (-1, 0)
        env.step((I, I, W))
        self.assertIsNotNone(env.state.buffers[0])
        self.assertIsNone(env.state.chefs[0].held)
        self.assertIsNone(env.state.chefs[1].held)

    def test_collision_does_not_freeze_unrelated_third_player(self):
        env = self.make()
        env.state.chefs[0].position = (2, 3)
        env.state.chefs[1].position = (4, 3)
        env.state.chefs[2].position = (8, 3)
        env.step((2, 3, 1))
        self.assertEqual([c.position for c in env.state.chefs], [(2, 3), (4, 3), (8, 4)])

    def test_swaps_and_blocked_chains_do_not_overlap(self):
        env = self.make()
        for i, position in enumerate(((2, 3), (3, 3), (4, 3))):
            env.state.chefs[i].position = position
        env.step((2, 3, W))
        self.assertEqual([c.position for c in env.state.chefs], [(2, 3), (3, 3), (4, 3)])
        env.step((2, 2, W))
        self.assertEqual([c.position for c in env.state.chefs], [(2, 3), (3, 3), (4, 3)])

    def test_inactive_player_is_not_a_static_obstacle(self):
        env = self.make(active_players=(0, 1))
        env.state.chefs[0].position = (5, 4) if (5, 4) in env.floor else (4, 5)
        env.state.chefs[2].position = (5, 5)
        env.step((2, W, 2))
        self.assertEqual(env.state.chefs[0].position, (5, 5))
        self.assertEqual(env.state.chefs[2].position, (5, 5))

    def test_all_geometries_remain_solvable_with_same_rules(self):
        for layout in ("compact", "offset", "bottleneck"):
            with self.subTest(layout=layout):
                env = self.make(layout=layout, max_batches=2)
                policy = ReferenceController()
                for _ in range(250):
                    _, _, done, _, _ = env.step(policy(env))
                    if done:
                        break
                self.assertEqual(env.state.metrics["normal_completed"], 2)
                self.assertEqual(env.state.metrics["joint_completed"], 2)

    def test_random_actions_preserve_invariants_on_all_maps(self):
        rng = np.random.default_rng(10)
        for layout in ("compact", "offset", "bottleneck"):
            for active in ((0, 1, 2), (0, 2), (1, 2)):
                env = self.make(layout=layout, active_players=active)
                for _ in range(300):
                    obs, _, done, _, _ = env.step(rng.integers(0, 6, 3))
                    env.validate_state()
                    self.assertTrue(all(np.isfinite(x).all() for x in obs))
                    if done:
                        break

    def test_snapshot_cannot_duplicate_food_or_corrupt_progress(self):
        env = self.make()
        saved = env.snapshot()
        corrupt = copy.deepcopy(saved)
        corrupt["state"]["chefs"][0]["held"] = {"kind": "ingredient", "order_id": env.state.orders[0].id, "certified": False}
        with self.assertRaises(AssertionError):
            env.restore(corrupt)
        self.assertEqual(env.snapshot(), saved)


if __name__ == "__main__":
    unittest.main()
