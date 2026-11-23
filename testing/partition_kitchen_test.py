import importlib.util
import os
import unittest

import numpy as np

from overcooked_ai_py.cooperative.partition import PartitionConfig, PartitionKitchen, TorchPartitionKitchen
from overcooked_ai_py.cooperative.partition.controller import ScriptedChefs
from overcooked_ai_py.cooperative.partition.model import (BEEF, CUT_POTATO, CUT_TOMATO, EAST, EGG,
    FIRST_MEAL, FLOOR, INTERACT, NORTH, POTATO, SOUTH, STAY, TOMATO, WEST, WIDTH, cell)


class PartitionRulesTest(unittest.TestCase):
    def setUp(self):
        self.env = PartitionKitchen()
        self.env.reset(11)

    def pose(self, cells, facing, hands):
        state = self.env.state
        state["pos"][:] = [cell(*xy) for xy in cells]
        state["facing"][:] = facing
        state["held"][:] = hands

    def step(self, *actions):
        return self.env.step(actions or [STAY] * 3)

    def test_paired_spawns_and_ticket_streams(self):
        a = self.env
        b = PartitionKitchen(PartitionConfig(layout="counter"))
        permutations = set()
        for seed in range(40):
            a.reset(seed)
            b.reset(seed)
            for k in ("pos", "facing", "ids", "recipe"):
                np.testing.assert_equal(a.state[k], b.state[k])
            pos = a.state["pos"]
            self.assertEqual(len(set(pos)), 3)
            self.assertTrue(all(p in FLOOR for p in pos))
            left = np.flatnonzero(pos % WIDTH < 6)
            self.assertEqual(len(left), 1)
            permutations.add(left.item())
            np.testing.assert_equal(a._batch.draws, b._batch.draws)
        self.assertEqual(permutations, {0, 1, 2})

    def test_chop_progress_interruption_and_relay(self):
        self.pose([(5, 4), (10, 1), (9, 2)], [WEST, SOUTH, EAST], [0, TOMATO, 0])
        self.step(STAY, INTERACT, STAY)
        self.assertEqual(self.env.state["cuts"][2], 0)
        self.step(STAY, INTERACT, INTERACT)
        self.assertEqual(self.env.state["cuts"][2], 1)
        self.step()
        self.assertEqual(self.env.state["cuts"][2], 0)
        self.step(STAY, STAY, INTERACT)
        self.step(STAY, INTERACT, STAY)
        self.step()
        self.step(STAY, STAY, INTERACT)
        self.assertEqual(self.env.state["held"][2], CUT_TOMATO)

    def test_pot_recipe_fork_and_tending(self):
        self.pose([(1, 1), (8, 1), (7, 2)], [NORTH, SOUTH, EAST], [0, CUT_POTATO, 0])
        self.step(STAY, INTERACT, STAY)
        self.assertEqual(self.env.state["locked"][1], -1)
        self.env.state["held"][1] = CUT_TOMATO
        self.step(STAY, INTERACT, STAY)
        self.env.state["held"][1] = EGG
        self.step(STAY, INTERACT, STAY)
        self.assertEqual(self.env.state["held"][1], EGG)
        self.env.state["held"][1] = BEEF
        self.step(STAY, INTERACT, STAY)
        self.step(STAY, STAY, INTERACT)
        self.assertEqual(self.env.state["locked"][1], 2)
        self.step()
        self.assertEqual(self.env.state["heat"][1], 0)
        self.assertEqual(self.env.state["locked"][1], 2)
        for i in range(4):
            self.step(STAY, INTERACT, INTERACT) if i % 2 else self.step(STAY, STAY, INTERACT)
            self.assertEqual(self.env.state["heat"][1], i + 1)
        self.step()
        self.step(STAY, INTERACT, STAY)
        self.assertEqual(self.env.state["held"][1], FIRST_MEAL + 2)
        self.assertEqual(self.env.state["mix"][1], 0)

    def test_last_tick_duplicates_refill_and_wrong_side(self):
        self.pose([(1, 3), (11, 3), (11, 5)], [WEST, EAST, WEST], [FIRST_MEAL, FIRST_MEAL, 0])
        s = self.env.state
        s["recipe"][:] = [0, 1, 0]
        s["due"][:] = [1, 1, 1]
        _, reward, _, _, info = self.step(INTERACT, INTERACT, STAY)
        self.assertEqual(reward, 20)
        np.testing.assert_equal(info["delivered"], [True, False, False])
        np.testing.assert_equal(info["expired"], [False, True, True])
        np.testing.assert_equal(self.env.state["ids"], [3, 4, 5])
        np.testing.assert_equal(self.env.state["due"] - 1, np.array(self.env.config.deadlines)[self.env.state["recipe"]])
        self.assertEqual(self.env.state["held"][0], FIRST_MEAL)

    def test_simultaneous_service_and_no_matching_ticket(self):
        self.pose([(1, 3), (11, 3), (11, 5)], [WEST, EAST, WEST], [FIRST_MEAL+1, FIRST_MEAL, 0])
        self.env.state["recipe"][:] = [2, 0, 1]
        _, reward, _, _, info = self.step(INTERACT, INTERACT, STAY)
        self.assertEqual(reward, 40)
        np.testing.assert_equal(info["delivered"], [False, True, True])
        np.testing.assert_equal(self.env.state["ids"], [0, 3, 4])
        self.env.state["held"][0] = FIRST_MEAL + 1
        self.env.state["recipe"][:] = 2
        _, reward, _, _, _ = self.step(INTERACT, STAY, STAY)
        self.assertEqual(reward, 0)
        self.assertEqual(self.env.state["held"][0], FIRST_MEAL + 1)

    def test_wall_modes_conflicts_and_material_types(self):
        for mode in ("sync", "counter"):
            self.env = PartitionKitchen(PartitionConfig(layout=mode))
            for y in range(1, 6):
                for item in (POTATO, CUT_POTATO, FIRST_MEAL + 1):
                    self.pose([(5, y), (7, y), (11, 1)], [EAST, WEST, NORTH], [item, 0, 0])
                    _, _, _, _, info = self.step(INTERACT, INTERACT, STAY)
                    if mode == "sync":
                        self.assertEqual(info["sync_successes"], y in (2, 4))
                        self.assertEqual(self.env.state["held"][1], item if y in (2, 4) else 0)
                    else:
                        self.assertEqual(self.env.state["held"][0], item)
                        self.step(INTERACT, STAY, STAY)
                        self.step()
                        _, _, _, _, info = self.step(STAY, INTERACT, STAY)
                        self.assertEqual(info["wait_steps"], 2)
                        self.assertEqual(self.env.state["held"][1], item)

    def test_parallel_api(self):
        from pettingzoo.test import parallel_api_test, parallel_seed_test
        from overcooked_ai_py.cooperative.partition.parallel import ParallelPartitionKitchen
        factory = lambda: ParallelPartitionKitchen(PartitionConfig(horizon=10))
        parallel_api_test(factory(), num_cycles=12)
        parallel_seed_test(factory, num_cycles=12)

    def test_all_six_scripted_deliveries(self):
        for layout in ("sync", "counter"):
            seen = set()
            for seed in range(4):
                env = PartitionKitchen(PartitionConfig(layout=layout))
                env.reset(seed)
                controller = ScriptedChefs(env)
                for _ in range(env.config.horizon):
                    _, _, _, _, info = env.step(controller.actions())
                    seen.update(info["recipe"][info["delivered"]])
                if len(seen) == 3:
                    break
            self.assertEqual(seen, {0, 1, 2})


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch dependency")
class PartitionTensorTest(unittest.TestCase):
    def test_per_tick_parity_and_masked_reset(self):
        import torch
        for layout in ("sync", "counter"):
            config = PartitionConfig(layout=layout, horizon=120)
            device = os.environ.get("OVERCOOKED_TEST_DEVICE", "cpu")
            batch = TorchPartitionKitchen(config, device=device, num_envs=4)
            native = [PartitionKitchen(config) for _ in range(4)]
            batch.reset(5)
            for i, env in enumerate(native):
                env.reset(5 + i)
            chefs = [ScriptedChefs(env) for env in native]
            rng = np.random.default_rng(9)
            for tick in range(config.horizon + 8):
                if tick == 75:
                    batch.reset(mask=batch.array([True, False, False, False]))
                    native[0].reset()
                    chefs[0] = ScriptedChefs(native[0])
                actions = [c.actions() if i != 3 else rng.integers(6, size=3) for i, c in enumerate(chefs)]
                obs, reward, term, trunc, info = batch.step(batch.array(np.asarray(actions)))
                self.assertEqual(obs.device, batch.device)
                for i, env in enumerate(native):
                    if env.state["tick"][0] >= config.horizon:
                        self.assertEqual(reward[i].item(), 0)
                        continue
                    expected, r, te, tr, details = env.step(actions[i])
                    np.testing.assert_allclose(obs[i].cpu(), expected, rtol=1e-6)
                    self.assertEqual((reward[i].item(), term[i].item(), trunc[i].item()), (r, te, tr))
                    for k, v in details.items():
                        np.testing.assert_equal(info[k][i].cpu().numpy(), v)
                for k, v in batch.state.items():
                    np.testing.assert_equal(v.cpu().numpy(), np.stack([e.state[k] for e in native]))
            before = batch.state["tick"].clone()
            with self.assertRaises(ValueError):
                batch.step(torch.full((4, 3), 6, device=device))
            torch.testing.assert_close(batch.state["tick"], before)
