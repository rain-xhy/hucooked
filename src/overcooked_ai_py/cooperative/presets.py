"""Training curricula and a separately frozen standard evaluation task."""

from dataclasses import replace

from .model import KitchenConfig


def preset(name):
    base = KitchenConfig()
    if name == "standard_eval":
        return base
    if name == "standard_train":
        return replace(base, shaping_scale=0.2)
    if name in ("handoff_1", "handoff_2", "handoff_4"):
        return replace(base, max_batches=1, sync_steps=int(name.rsplit("_", 1)[1]),
                       duration_jitter=0, reset_mode="handoff_ready", shaping_scale=0.2)
    if name == "kitchen_ready":
        return replace(base, max_batches=1, reset_mode="kitchen_ready", shaping_scale=0.2)
    if name == "full_easy":
        return replace(base, normal_deadline=180, joint_deadline=200,
                       sync_deadline=160, duration_jitter=0, shaping_scale=0.2)
    raise ValueError("Unknown preset: " + name)
