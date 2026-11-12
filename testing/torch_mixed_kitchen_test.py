"""Compare batched PyTorch transitions and the Parallel API with native MixedKitchen."""

import importlib.util
import os
import unittest
from dataclasses import replace

import numpy as np

from overcooked_ai_py.cooperative import KitchenConfig, MixedKitchen
from overcooked_ai_py.cooperative.controller import ReferenceController
from overcooked_ai_py.cooperative.kitchen import INTERACT, STAY
from overcooked_ai_py.cooperative.parallel import ParallelKitchen
from overcooked_ai_py.cooperative.presets import preset


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch dependency")
class TestTorchMixedKitchen(unittest.TestCase):
    def setUp(self):
        import torch

        from overcooked_ai_py.cooperative.torch_kitchen import TorchMixedKitchen

        self.torch = torch
        self.backend = TorchMixedKitchen
        self.device = torch.empty(0, device=os.getenv("OVERCOOKED_TEST_DEVICE", "cpu")).device

    def assert_observations(self, actual, expected):
        self.assertEqual(actual.device, self.device)
        np.testing.assert_array_equal(actual.cpu().numpy(), np.asarray(expected))

    def assert_step(self, backend, result, lane, reference, expected):
        obs, reward, terminated, truncated, info = result
        native_obs, native_reward, native_terminated, native_truncated, native_info = expected
        self.assert_observations(obs[lane], native_obs)
        self.assertEqual(reward.device, self.device)
        self.assertAlmostEqual(reward[lane].item(), native_reward)
        self.assertEqual(terminated[lane].item(), native_terminated)
        self.assertEqual(truncated[lane].item(), native_truncated)
        native_shaping = native_info["shaping_reward"]
        actual_info = info.to_native(lane)
        self.assertAlmostEqual(actual_info["shaping_reward"], native_shaping)
        actual_info["shaping_reward"] = native_shaping
        self.assertEqual(actual_info, native_info)
        self.assertEqual(backend.snapshot(lane), reference.snapshot())

    def test_batched_reference_rollouts_on_all_geometries(self):
        for layout in ("compact", "offset", "bottleneck"):
            with self.subTest(layout=layout):
                config = replace(preset("standard_train"), layout=layout, max_batches=2)
                backend = self.backend(config, device=self.device, num_envs=2)
                references = [MixedKitchen(config) for _ in range(2)]
                policies = [ReferenceController() for _ in references]
                obs, info = backend.reset(seed=12)
                for lane, env in enumerate(references):
                    expected, expected_info = env.reset(seed=12 + lane)
                    self.assert_observations(obs[lane], expected)
                    self.assertEqual(info.to_native(lane), expected_info)
                while not all(ref.state.terminated for ref in references):
                    active = [not ref.state.terminated for ref in references]
                    actions = [policy(ref) if live else [STAY] * 3
                               for ref, policy, live in zip(references, policies, active)]
                    actual = backend.step(self.torch.tensor(actions, device=self.device))
                    for lane, (ref, live) in enumerate(zip(references, active)):
                        if live:
                            self.assert_step(backend, actual, lane, ref, ref.step(actions[lane]))
                        else:
                            self.assertEqual(backend.snapshot(lane), ref.snapshot())
                for ref in references:
                    self.assertEqual(ref.state.metrics["joint_completions"], 2)

    def test_prepared_reset_abort_expiry_and_selective_replay(self):
        config = replace(preset("handoff_4"), horizon=12, sync_deadline=6, joint_deadline=10,
                         normal_deadline=9, deadline_jitter=1, active_players=(2, 0))
        native = MixedKitchen(config)
        native.reset(seed=28)
        backend = self.backend(config, device=self.device, num_envs=2)
        backend.restore(native.snapshot(), lane=1)
        lane_zero = backend.snapshot(0)
        while not native.state.terminated:
            tick = native.state.tick
            action = (INTERACT, STAY, INTERACT) if tick < 2 else (STAY,) * 3
            before = native.snapshot()
            if tick == 1:
                backend.restore(before, lane=1)
            result = backend.step([[STAY] * 3, action])
            self.assert_step(backend, result, 1, native, native.step(action))
        self.assertEqual(native.state.metrics["joint_aborts"], 1)
        self.assertEqual(native.state.metrics["joint_missed"], 1)
        finished = backend.snapshot(1)
        obs, _ = backend.reset(seed=0, mask=[True, False])
        self.assertEqual(backend.snapshot(0), lane_zero)
        self.assertEqual(backend.snapshot(1), finished)
        self.assert_observations(obs[:, 0], backend.centralized_state().cpu().numpy())

    def test_random_actions_and_unseeded_reset_preserve_native_replay(self):
        config = replace(preset("kitchen_ready"), max_batches=3, active_players=(1, 0, 2),
                         horizon=50, normal_deadline=12, sync_deadline=9, joint_deadline=17,
                         expiry_penalty=-2.3, batch_bonus=0.3)
        backend = self.backend(config, device=self.device, num_envs=2)
        references = [MixedKitchen(config), MixedKitchen(config)]
        rng = np.random.default_rng(19)
        for lane, native in enumerate(references):
            native.reset(seed=lane)
        for _ in range(65):
            ended = [native.state.terminated for native in references]
            if any(ended):
                obs, _ = backend.reset(mask=ended)
                for lane, (native, reset) in enumerate(zip(references, ended)):
                    if reset:
                        expected, _ = native.reset()
                        self.assert_observations(obs[lane], expected)
            actions = rng.integers(0, 6, size=(2, 3))
            result = backend.step(actions)
            for lane, native in enumerate(references):
                self.assert_step(backend, result, lane, native, native.step(actions[lane]))

    def test_parallel_interface_matches_native_dicts(self):
        config = replace(preset("handoff_1"), horizon=5, active_players=(2, 0))
        native = ParallelKitchen(config)
        actual = ParallelKitchen(config, device=self.device, render_mode="rgb_array")
        obs, info = actual.reset(seed=7, options={})
        expected, expected_info = native.reset(seed=7, options={})
        self.assertEqual(actual.agents, native.agents)
        self.assertEqual(info, expected_info)
        for agent in actual.agents:
            self.assertTrue(actual.observation_space(agent).contains(obs[agent]))
            np.testing.assert_array_equal(obs[agent], expected[agent])
        self.assertTrue(actual.state_space.contains(actual.state()))
        self.assertEqual(actual.render().ndim, 3)
        with self.assertRaises(ValueError):
            actual.step({"chef_0": STAY})
        with self.assertRaises(ValueError):
            actual.step({agent: True for agent in actual.agents})
        for tick in range(config.horizon):
            actions = {agent: INTERACT if tick == 0 else STAY for agent in actual.agents}
            result, expected = actual.step(actions), native.step(actions)
            for agent, array in result[0].items():
                np.testing.assert_array_equal(array, expected[0][agent])
            self.assertEqual(result[1:], expected[1:])
        self.assertEqual(actual.step({}), ({}, {}, {}, {}, {}))
        with self.assertRaises(RuntimeError):
            actual.step({"chef_0": STAY})

    @unittest.skipUnless(importlib.util.find_spec("pettingzoo"), "optional PettingZoo not installed")
    def test_official_parallel_api(self):
        from pettingzoo.test import parallel_api_test

        for active in ((0, 1, 2), (0, 2)):
            env = ParallelKitchen(KitchenConfig(active_players=active, horizon=12), device=self.device)
            parallel_api_test(env, num_cycles=24)
