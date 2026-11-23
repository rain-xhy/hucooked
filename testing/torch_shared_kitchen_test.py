import importlib.util
import os
import unittest

import numpy as np

from overcooked_ai_py.cooperative.shared import INTERACT, STAY, SharedKitchen, SharedKitchenConfig
from overcooked_ai_py.cooperative.shared.controller import ScriptedController


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch dependency")
class TorchSharedKitchenTest(unittest.TestCase):
    def backend(self, config, lanes):
        from overcooked_ai_py.cooperative.shared.torch_kitchen import TorchSharedKitchen
        return TorchSharedKitchen(config, num_envs=lanes, device=os.getenv("OVERCOOKED_TEST_DEVICE", "cpu"))

    def compare(self, native, obs):
        self.assertEqual(obs.device, self.gpu.device)
        np.testing.assert_allclose(obs.cpu().numpy(), native, rtol=1e-6, atol=1e-7)

    def test_concurrent_unload_chop_and_full_mixed_episode(self):
        import torch
        config = SharedKitchenConfig(normal_orders=1, joint_orders=2)
        self.gpu = self.backend(config, 4)
        envs = [SharedKitchen(config) for _ in range(4)]
        obs, _ = self.gpu.reset(seed=17)
        for i, env in enumerate(envs):
            self.compare(env.reset(17 + i)[0], obs[i])
        controllers = [ScriptedController(env) for env in envs]
        total = [0.] * 4
        witnessed = False
        for tick in range(config.horizon):
            actions = []
            for i, (env, controller) in enumerate(zip(envs, controllers)):
                if env.terminated or env.truncated or (i == 1 and tick < 5):
                    actions.append([STAY] * 3)
                    continue
                joint = controller.actions(env)
                cooker = env.state.cooker
                if (env.state.cutting.item and env.state.cutting.progress == 0 and cooker.item
                        and env.order(cooker.item.order_id).kind == "joint"):
                    both = cooker.remaining == 0 and all(joint[j] == INTERACT for j in (controller.main, controller.server))
                    joint[controller.side] = INTERACT if both else STAY
                actions.append(joint)
            obs, rewards, terms, truncs, info = self.gpu.step(torch.tensor(actions, device=self.gpu.device))
            for i, env in enumerate(envs):
                if env.terminated:
                    expected, reward, term, trunc = env.observations(), 0., True, False
                else:
                    expected, reward, term, trunc, detail = env.step(actions[i])
                    events = detail["events"]
                    unload = next((e for e in events if e["type"] == "unload"), None)
                    joint_unload = bool(unload and env.order(unload["order_id"]).kind == "joint")
                    chopped = any(e["type"] == "chop" for e in events)
                    self.assertEqual(info["joint_unload"][i].item(), joint_unload)
                    self.assertEqual(info["chopped"][i].item(), chopped)
                    if unload:
                        self.assertEqual(info["unloaders"][i].nonzero().flatten().tolist(), unload["actors"])
                    witnessed |= joint_unload and chopped
                self.compare(expected, obs[i])
                self.assertEqual((rewards[i].item(), terms[i].item(), truncs[i].item()), (reward, term, trunc))
                total[i] += reward
            if all(env.terminated for env in envs):
                break
        self.assertEqual(total, [60.] * 4)
        self.assertTrue(witnessed)

    def test_random_actions_inactive_slots_and_masked_reset(self):
        import torch
        config = SharedKitchenConfig(active_players=(0, 2), horizon=30, active_limit=1)
        self.gpu = self.backend(config, 6)
        envs = [SharedKitchen(config) for _ in range(6)]
        self.gpu.reset(seed=7)
        for i, env in enumerate(envs):
            env.reset(7 + i)
        rng = np.random.default_rng(48)
        for tick in range(42):
            actions = rng.integers(6, size=(6, 3))
            actions[:, 1] = STAY
            if tick == 20:
                mask = torch.arange(6, device=self.gpu.device) % 2 == 0
                obs, _ = self.gpu.reset(mask=mask)
                for i, env in enumerate(envs):
                    expected = env.reset()[0] if i % 2 == 0 else env.observations()
                    self.compare(expected, obs[i])
            obs, rewards, terms, truncs, _ = self.gpu.step(actions)
            for i, env in enumerate(envs):
                if env.truncated:
                    expected, reward, term, trunc = env.observations(), 0., False, True
                else:
                    expected, reward, term, trunc, _ = env.step(actions[i])
                self.compare(expected, obs[i])
                self.assertEqual((rewards[i].item(), terms[i].item(), truncs[i].item()), (reward, term, trunc))
        before = self.gpu.observations()
        actions[:, 1] = INTERACT
        with self.assertRaises(ValueError):
            self.gpu.step(actions)
        self.assertTrue(torch.equal(before, self.gpu.observations()))
