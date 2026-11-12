"""Controlled capability probes; these are not the standard reset distribution."""

from collections import deque

from .kitchen import DIRECTIONS, INTERACT, STAY, MixedKitchen, adjacent
from .model import Item, KitchenConfig, Pot


def overlap_probe(active_players=(0, 1, 2)):
    """Two-chef upper bound 25; a three-chef schedule constructively scores 60.

    Certification needs both chefs on all first 20 steps. The ordinary dish
    must be served by step 20. With two chefs, even one ordinary action makes
    certification impossible. Penalties and the two rewards give an upper
    bound max(30-5, 20-5, -10)=25. A third chef can serve concurrently.
    """
    config = KitchenConfig(horizon=50, max_batches=1, sync_steps=20,
                           normal_deadline=20, joint_deadline=40, sync_deadline=20,
                           deadline_jitter=0, duration_jitter=0, shuffle_players=False,
                           active_players=active_players)
    env = MixedKitchen(config)
    s = env.state
    s.chefs[0].position, s.chefs[0].orientation = (4, 3), (1, 0)
    s.chefs[1].position, s.chefs[1].orientation = (6, 3), (-1, 0)
    s.chefs[2].position, s.chefs[2].orientation = (6, 4), (-1, 0)
    s.orders[0].stock = s.orders[1].stock = 0
    s.pots[1] = Pot(s.orders[0].id, 0)
    s.buffers[1] = Item("dish")
    s.chefs[0].held = Item("soup", s.orders[1].id)
    env.validate_state()

    def go(i, target, facing):
        chef = env.state.chefs[i]
        if chef.position == target:
            return INTERACT if chef.orientation == facing else DIRECTIONS.index(facing)
        blocked = {c.position for j, c in enumerate(env.state.chefs) if j in env.active and j != i}
        queue, seen = deque([(chef.position, None)]), {chef.position}
        while queue:
            position, first = queue.popleft()
            if position == target:
                return first
            for action, direction in enumerate(DIRECTIONS):
                neighbor = adjacent(position, direction)
                if neighbor in env.floor and neighbor not in blocked and neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, action if first is None else first))
        return STAY

    for _ in range(50):
        actions = [INTERACT if env.state.tick < 20 else STAY] * 2 + [STAY]
        if env.state.tick >= 20 and env.state.chefs[1].held is not None:
            actions[1] = go(1, (9, 5), (1, 0))
        if 2 in env.active:
            chef = env.state.chefs[2]
            if env.state.orders[0].status == "active":
                if chef.held is None:
                    actions[2] = go(2, (6, 4), (-1, 0))
                elif chef.held.kind == "dish":
                    actions[2] = go(2, (8, 1), (0, -1))
                else:
                    actions[2] = go(2, (9, 5), (1, 0))
            elif chef.position != (9, 3):
                # Clear S so the cooperative dish can be delivered afterwards.
                actions[2] = go(2, (9, 3), (1, 0))
        _, _, done, _, _ = env.step(actions)
        if done:
            break
    return {"score": env.state.score, "metrics": env.state.metrics,
            "steps": env.state.tick, "two_chef_upper_bound": 25.0,
            "scenario": "controlled_overlap_not_standard_reset"}


def single_category_score_bounds(config):
    """Optimistic upper bounds, even allowing instant preparation and delivery."""
    def batches_before_timeout(deadline):
        minimum_period = deadline - config.deadline_jitter + 1
        return min(config.max_batches, 1 + (config.horizon - 1) // minimum_period)
    return {
        "normal_only": batches_before_timeout(config.sync_deadline) * config.normal_reward,
        "joint_only": batches_before_timeout(config.normal_deadline) * config.joint_reward,
        "assumptions": "Ignored category is never delivered; missed-order penalties are relaxed to zero; no batch bonus can be earned.",
    }
