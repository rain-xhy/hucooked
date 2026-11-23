from .engine import BatchKitchen
from .model import PartitionConfig


class PartitionKitchen:
    """Single native NumPy kitchen. State arrays use the chef and facility order in model.py."""

    def __init__(self, config=None):
        self._batch = BatchKitchen(config, 1)
        self.config = self._batch.config

    @property
    def state(self):
        return {key: value[0] for key, value in self._batch.state.items()}

    def observations(self):
        return self._batch.observations()[0]

    def centralized_state(self):
        return self.observations()[0, :-1]

    def reset(self, seed=None):
        obs, info = self._batch.reset(seed)
        return obs[0], info

    def step(self, actions):
        if self.state["tick"][0] >= self.config.horizon:
            raise RuntimeError("Reset before stepping a finished episode")
        obs, reward, terminated, truncated, info = self._batch.step([actions])
        return obs[0], float(reward[0]), bool(terminated[0]), bool(truncated[0]), {
            k: v[0] for k, v in info.items()}


class TorchPartitionKitchen(BatchKitchen):
    def __init__(self, config=None, *, device="cuda", num_envs=1):
        super().__init__(config, num_envs, device=device)


__all__ = ["PartitionConfig", "PartitionKitchen", "TorchPartitionKitchen"]
