"""Batched PyTorch execution of native OvercookedGridworld layouts."""

from typing import NamedTuple

import torch

from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import (
    EVENT_TYPES,
    ObjectState,
    OvercookedState,
    PlayerState,
    Recipe,
    SoupState,
)


FLOOR, COUNTER, ONION_SOURCE, TOMATO_SOURCE, DISH_SOURCE, POT, SERVING = range(7)
EMPTY, ONION, TOMATO, DISH, SOUP = range(5)
ITEM_NAMES = (None, "onion", "tomato", "dish", "soup")
KIND, COOK_TICK, COOK_OVERRIDE, INGREDIENTS = range(4)
INTERACT_INDEX = Action.ACTION_TO_INDEX[Action.INTERACT]


class TorchOvercookedState(NamedTuple):
    """Batch of states. Positions are flattened as y * width + x.

    Object records contain kind, cooking tick, optional time override (-1
    when absent), and the ingredients in insertion order (zero padded).
    """

    positions: torch.Tensor
    orientations: torch.Tensor
    objects: torch.Tensor
    held_objects: torch.Tensor
    timestep: torch.Tensor


class TorchOvercookedEnv:
    """A batch of independent kitchens sharing one native MDP configuration."""

    def __init__(self, mdp, horizon, device="cuda", num_envs=1, start_state_fn=None):
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.mdp = mdp
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.num_players = mdp.num_players
        self.horizon = horizon
        self.width, self.height = mdp.shape
        self.start_state_fn = start_state_fn

        Recipe.configure(mdp.recipe_config)
        self._max_ingredients = Recipe.MAX_NUM_INGREDIENTS
        self._recipes = [(onions, tomatoes, Recipe(["onion"] * onions + ["tomato"] * tomatoes))
                         for onions in range(self._max_ingredients + 1)
                         for tomatoes in range(self._max_ingredients - onions + 1) if onions + tomatoes]
        self._empty_item = [EMPTY, -1, -1, *([EMPTY] * self._max_ingredients)]
        self._tiles = self._tensor([" XOTDPS".index(tile) for row in mdp.terrain_mtx for tile in row])
        self._cells = torch.arange(self.width * self.height, device=self.device)
        self._rows = torch.arange(num_envs, device=self.device)
        self._directions = torch.arange(len(Direction.ALL_DIRECTIONS), device=self.device)
        xy = torch.stack((self._cells % self.width, self._cells // self.width), dim=-1)
        neighbors = xy[:, None] + self._tensor(Action.MOTION_ACTIONS)
        neighbors = neighbors.clamp_min(0).minimum(self._tensor((self.width - 1, self.height - 1)))
        self._neighbors = neighbors[..., 0] + self.width * neighbors[..., 1]
        self._pair_a, self._pair_b = torch.triu_indices(self.num_players, self.num_players, 1, device=self.device)
        self._static_layers = self._tiles == self._tensor((POT, COUNTER, ONION_SOURCE, TOMATO_SOURCE,
                                                         DISH_SOURCE, SERVING))[:, None]
        self._cook_times = self._recipe_matrix(lambda _onions, _tomatoes, recipe: recipe.time)
        self._shaping = self._tensor(
            [mdp.reward_shaping_params[key] for key in (
                "PLACEMENT_IN_POT_REW", "DISH_PICKUP_REWARD", "SOUP_PICKUP_REWARD",
            )], dtype=torch.float64,
        )
        self._empty = self._tensor(self._empty_item, dtype=torch.float64)
        self._initial, self._initial_rewards, self._initial_optimal, orders = self._encode_start(mdp.get_standard_start_state())
        self._orders = [orders] * num_envs
        self.state = TorchOvercookedState(*(self._repeat(field) for field in self._initial))
        self._rewards = self._repeat(self._initial_rewards)
        self._optimal = self._repeat(self._initial_optimal)
        self._sparse_total = torch.zeros((num_envs, self.num_players), dtype=torch.float64, device=self.device)
        self._shaped_total = torch.zeros_like(self._sparse_total)
        if start_state_fn is not None:
            self.reset()

    def _tensor(self, data, dtype=torch.long):
        return torch.tensor(data, dtype=dtype, device=self.device)

    def _repeat(self, tensor):
        return tensor.unsqueeze(0).expand(self.num_envs, *tensor.shape)

    def _flat(self, position):
        return position[0] + position[1] * self.width

    def _recipe_matrix(self, evaluate):
        matrix = [[0.0] * (self._max_ingredients + 1) for _ in range(self._max_ingredients + 1)]
        for onions, tomatoes, recipe in self._recipes:
            matrix[onions][tomatoes] = evaluate(onions, tomatoes, recipe)
        return self._tensor(matrix, dtype=torch.float64)

    def _encode_item(self, obj):
        data = self._empty_item.copy()
        if obj is None:
            return data
        data[KIND] = ITEM_NAMES.index(obj.name)
        if obj.name == "soup":
            data[COOK_TICK] = obj._cooking_tick
            if obj._cook_time is not None:
                data[COOK_OVERRIDE] = obj._cook_time
            for index, name in enumerate(obj.ingredients, INGREDIENTS):
                data[index] = ITEM_NAMES.index(name)
        return data

    def _encode_start(self, native):
        objects = [self._empty_item] * (self.width * self.height)
        for position, item in native.objects.items():
            objects[self._flat(position)] = self._encode_item(item)
        state = TorchOvercookedState(
            self._tensor([self._flat(p.position) for p in native.players]),
            self._tensor([Direction.DIRECTION_TO_INDEX[p.orientation] for p in native.players]),
            self._tensor(objects, dtype=torch.float64),
            self._tensor([self._encode_item(p.held_object) for p in native.players], dtype=torch.float64),
            self._tensor(native.timestep),
        )
        values = {(onions, tomatoes): self.mdp.get_recipe_value(native, recipe)
                  for onions, tomatoes, recipe in self._recipes}
        rewards = self._recipe_matrix(lambda onions, tomatoes, _recipe: values[onions, tomatoes])
        optimal = [[0.0] * (self._max_ingredients + 1) for _ in range(self._max_ingredients + 1)]
        for onions, tomatoes in [(0, 0), *values]:
            best = max(value for (o, t), value in values.items() if o >= onions and t >= tomatoes)
            optimal[onions][tomatoes] = best if best > 0 else values.get((onions, tomatoes), 0)
        orders = tuple(tuple(order.to_dict() for order in group) for group in (native.bonus_orders, native.all_orders))
        return state, rewards, self._tensor(optimal, dtype=torch.float64), orders

    def reset(self, mask=None):
        """Reset selected lanes. Default starts and tensor masks stay on device."""
        if mask is None:
            mask = torch.ones_like(self.state.timestep, dtype=torch.bool)
        else:
            mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            if mask.shape != (self.num_envs,):
                raise ValueError(f"Expected reset mask of shape {(self.num_envs,)}")
        if self.start_state_fn is None:
            self.state = TorchOvercookedState(*(
                torch.where(mask.reshape((-1,) + (1,) * template.ndim), template, old)
                for template, old in zip(self._initial, self.state)
            ))
        else:
            # A Python start-state callback is the host boundary for custom starts.
            indices = mask.nonzero().flatten().tolist()
            if indices:
                Recipe.configure(self.mdp.recipe_config)
                rows = self._tensor(indices)
                starts, rewards, optimal, orders = zip(*(self._encode_start(self.start_state_fn()) for _ in indices))
                self.state = TorchOvercookedState(*(
                    old.index_copy(0, rows, torch.stack(new)) for old, new in zip(self.state, zip(*starts))
                ))
                self._rewards = self._rewards.index_copy(0, rows, torch.stack(rewards))
                self._optimal = self._optimal.index_copy(0, rows, torch.stack(optimal))
                for lane, order in zip(indices, orders):
                    self._orders[lane] = order
        self._sparse_total = self._sparse_total.masked_fill(mask[:, None], 0)
        self._shaped_total = self._shaped_total.masked_fill(mask[:, None], 0)
        return self.state

    def is_done(self):
        return self.state.timestep >= self.horizon

    @staticmethod
    def _ingredient_counts(items):
        ingredients = items[..., INGREDIENTS:]
        return (ingredients == ONION).sum(dim=-1), (ingredients == TOMATO).sum(dim=-1)

    def _cook_time(self, items):
        onions, tomatoes = self._ingredient_counts(items)
        override = items[..., COOK_OVERRIDE]
        return torch.where(override >= 0, override, self._cook_times[onions, tomatoes])

    def _interact(self, player, target, interact, objects, held, full_pots, nonempty_pots):
        tile = self._tiles[target]
        counter_item = objects[self._rows, target]
        item = held[:, player]
        kind, target_kind = item[:, KIND], counter_item[:, KIND]
        empty = kind == EMPTY
        counter = interact & (tile == COUNTER)
        take = counter & empty & (target_kind != EMPTY)
        drop = counter & ~empty & (target_kind == EMPTY)
        onion = interact & empty & (tile == ONION_SOURCE)
        tomato = interact & empty & (tile == TOMATO_SOURCE)
        dish = interact & empty & (tile == DISH_SOURCE)
        at_pot = interact & (tile == POT)
        onions, tomatoes = self._ingredient_counts(counter_item)
        count = onions + tomatoes
        tick = counter_item[:, COOK_TICK]
        add = at_pot & ((kind == ONION) | (kind == TOMATO)) & (count < self._max_ingredients) & (tick < 0)
        start = at_pot & empty & (count > 0) & (tick < 0) & (not self.mdp.old_dynamics)
        plate = at_pot & (kind == DISH) & (target_kind == SOUP) & (tick >= 0) & (tick >= self._cook_time(counter_item))
        serve = interact & (tile == SERVING) & (kind == SOUP)

        useful_dish = torch.zeros_like(interact)
        if self.num_players == 2:
            useful_dish = ((held[..., KIND] == DISH).sum(dim=1) < nonempty_pots)
            useful_dish &= ~((objects[..., KIND] == DISH) & (self._tiles == COUNTER)).any(dim=1)

        held_onions, held_tomatoes = self._ingredient_counts(item)
        sparse = torch.where(serve, self._rewards[self._rows, held_onions, held_tomatoes], 0)
        shaped = torch.where(torch.stack((add, dish & useful_dish, plate), dim=1), self._shaping, 0).sum(dim=1)

        flags = {"onion_pickup": onion | (take & (target_kind == ONION)),
                 "tomato_pickup": take & (target_kind == TOMATO),
                 "dish_pickup": dish | (take & (target_kind == DISH)),
                 "soup_pickup": plate | (take & (target_kind == SOUP)), "soup_delivery": serve}
        for i, name in enumerate(ITEM_NAMES[ONION:], ONION):
            flags[f"{name}_drop"] = drop & (kind == i)
        if self.num_players == 2:
            peer_kind = held[:, 1 - player, KIND]
            all_full = (full_pots | (self._tiles != POT)).all(dim=1)
            for name in ("onion", "tomato"):
                flags[f"useful_{name}_pickup"] = flags[f"{name}_pickup"] & (~all_full | (peer_kind == DISH))
                flags[f"useful_{name}_drop"] = flags[f"{name}_drop"] & all_full & (peer_kind != DISH)
            flags["useful_dish_pickup"] = flags["dish_pickup"] & useful_dish
            flags["useful_dish_drop"] = flags["dish_drop"] & ~full_pots.any(dim=1) & (peer_kind != ONION)

        old_value = self._optimal[self._rows, onions, tomatoes]
        new_value = self._optimal[self._rows, (onions + (kind == ONION)).clamp_max(self._max_ingredients),
                                  (tomatoes + (kind == TOMATO)).clamp_max(self._max_ingredients)]
        for ingredient, name in ((ONION, "onion"), (TOMATO, "tomato")):
            potted = add & (kind == ingredient)
            flags[f"potting_{name}"] = potted
            flags[f"optimal_{name}_potting"] = potted & (old_value == new_value)
            flags[f"viable_{name}_potting"] = potted & (new_value > 0)
            flags[f"catastrophic_{name}_potting"] = potted & (old_value > 0) & (new_value == 0)
            flags[f"useless_{name}_potting"] = potted & (old_value == 0)

        # Assign both sides of the interaction only after reading the old item records.
        next_held = torch.where((take | plate)[:, None], counter_item, item)
        next_held = torch.where((drop | add | serve)[:, None], self._empty, next_held)
        next_held[:, KIND] = torch.where(onion, ONION, torch.where(tomato, TOMATO,
                                             torch.where(dish, DISH, next_held[:, KIND])))
        next_item = torch.where(drop[:, None], item, counter_item)
        next_item = torch.where((take | plate)[:, None], self._empty, next_item)
        next_item[:, KIND] = torch.where(add, SOUP, next_item[:, KIND])
        next_item[:, COOK_TICK] = torch.where(start, 0, next_item[:, COOK_TICK])
        ingredient_slot = INGREDIENTS + count.clamp_max(self._max_ingredients - 1)
        next_item[self._rows, ingredient_slot] = torch.where(add, kind, next_item[self._rows, ingredient_slot])
        held[:, player] = next_held
        objects[self._rows, target] = next_item
        return sparse, shaped, flags

    @torch.no_grad()
    def step(self, actions):
        """Advance all unfinished lanes with native Action indices [batch, players]."""
        actions = torch.as_tensor(actions, device=self.device)
        if actions.shape != (self.num_envs, self.num_players):
            raise ValueError(f"Expected action shape {(self.num_envs, self.num_players)}")
        if actions.dtype not in (torch.int32, torch.int64):
            raise TypeError("Action indices must be integers")
        if self.device.type == "cpu" and ((actions < 0) | (actions >= Action.NUM_ACTIONS)).any():
            raise ValueError("Action indices are outside the native action space")
        positions, orientations, objects, held, timestep = self.state
        active = ~self.is_done()
        objects, held = objects.clone(), held.clone()
        # Native SoupState.deepcopy clears cook-time overrides on transition.
        for items in (objects, held):
            items[..., COOK_OVERRIDE].masked_fill_(active[:, None], -1)

        pot_mask = self._tiles == POT
        pot_count = (objects[..., INGREDIENTS:] != EMPTY).sum(dim=-1)
        cooking = objects[..., COOK_TICK] >= 0
        full_pots = pot_mask & (cooking | (pot_count == self._max_ingredients))
        nonempty_pots = (pot_mask & (cooking | ((pot_count > 0) & ~full_pots))).sum(dim=1)
        target = self._neighbors[positions, orientations]
        player_results = [self._interact(player, target[:, player], active & (actions[:, player] == INTERACT_INDEX),
                                        objects, held, full_pots, nonempty_pots)
                          for player in range(self.num_players)]

        moving = active[:, None] & (actions >= 0) & (actions < len(Direction.ALL_DIRECTIONS))
        orientations = torch.where(moving, actions, orientations)
        destination = self._neighbors[positions, orientations]
        proposed = torch.where(moving & (self._tiles[destination] == FLOOR), destination, positions)
        a, b = self._pair_a, self._pair_b
        collision = (proposed[:, a] == proposed[:, b]) | ((proposed[:, a] == positions[:, b]) &
                                                          (proposed[:, b] == positions[:, a]))
        positions = torch.where(collision.any(dim=1, keepdim=True), positions, proposed)

        tick = objects[..., COOK_TICK]
        if self.mdp.old_dynamics:
            full = (objects[..., INGREDIENTS:] != EMPTY).sum(dim=-1) == 3
            tick.masked_fill_(active[:, None] & (objects[..., KIND] == SOUP) & (tick < 0) & full, 0)
        tick.add_(active[:, None] & (tick >= 0) & (tick < self._cook_time(objects)))
        self.state = TorchOvercookedState(positions, orientations, objects, held, timestep + active)

        sparse, shaped, events = zip(*player_results)
        sparse, shaped = torch.stack(sparse, dim=1), torch.stack(shaped, dim=1)
        self._sparse_total = self._sparse_total + sparse
        self._shaped_total = self._shaped_total + shaped
        false = torch.zeros_like(active)
        info = {"sparse_r_by_agent": sparse, "shaped_r_by_agent": shaped,
                "events": {key: torch.stack([event.get(key, false) for event in events], dim=1) for key in EVENT_TYPES},
                "episode": {"ep_sparse_r": self._sparse_total.sum(dim=1), "ep_shaped_r": self._shaped_total.sum(dim=1),
                            "ep_sparse_r_by_agent": self._sparse_total, "ep_shaped_r_by_agent": self._shaped_total,
                            "ep_length": self.state.timestep}}
        return self.state, sparse.sum(dim=1), self.is_done(), info

    def to_native_state(self, lane=0):
        """Copy one lane to the CPU for existing agents, debugging or rendering."""
        Recipe.configure(self.mdp.recipe_config)
        positions, orientations, objects, held, timestep = (field[lane].tolist() for field in self.state)

        def xy(position):
            return position % self.width, position // self.width

        def decode(position, data):
            kind, tick, override, *ingredients = data
            name = ITEM_NAMES[int(kind)]
            if kind == SOUP:
                return SoupState(position, [ObjectState(ITEM_NAMES[int(i)], position) for i in ingredients if i],
                                 int(tick), None if override < 0 else override)
            return ObjectState(name, position) if kind else None

        players = [PlayerState(xy(p), Direction.INDEX_TO_DIRECTION[d], decode(xy(p), item))
                   for p, d, item in zip(positions, orientations, held)]
        board = {xy(i): decode(xy(i), item) for i, item in enumerate(objects) if item[KIND] != EMPTY}
        bonus, allowed = self._orders[lane]
        return OvercookedState(players, board, bonus_orders=bonus, all_orders=allowed, timestep=timestep)

    def lossless_state_encoding(self):
        """Native two-chef lossless grid as [batch, player, width, height, 26]."""
        if self.num_players != 2:
            raise ValueError("Native lossless encoding requires exactly two players")
        positions, orientations, objects, held, timestep = self.state
        occupancy = positions[..., None] == self._cells
        facing = orientations[..., None] == self._directions
        headings = occupancy[:, :, None] & facing[..., None]
        static = self._static_layers.unsqueeze(0).expand(self.num_envs, -1, -1)

        def layers(items, in_pot):
            kind, tick = items[..., KIND], items[..., COOK_TICK]
            soup = kind == SOUP
            idle = soup & in_pot & (tick < 0)
            cooking = soup & in_pot & ~idle
            remaining = self._cook_time(items) - tick
            ready = soup & (~in_pot | (cooking & (remaining <= 0)))
            onions, tomatoes = self._ingredient_counts(items)
            return torch.stack((onions * idle, tomatoes * idle, onions * (soup & ~idle), tomatoes * (soup & ~idle),
                                torch.where(cooking, remaining, 0), ready, kind == DISH, kind == ONION,
                                kind == TOMATO), dim=1).long()

        dynamic = layers(objects, self._tiles == POT)
        held_in_pot = torch.zeros_like(positions, dtype=torch.bool)
        dynamic.scatter_add_(2, positions[:, None].expand(-1, dynamic.shape[1], -1), layers(held, held_in_pot))
        urgency = (self.horizon - timestep < 40)[:, None, None].expand(-1, 1, len(self._cells))
        common = torch.cat((static, dynamic, urgency), dim=1)
        views = torch.stack([torch.cat((p, h.flatten(1, 2), common), dim=1)
                             for p, h in ((occupancy, headings), (occupancy.flip(1), headings.flip(1)))], dim=1)
        return views.reshape(self.num_envs, 2, 26, self.height, self.width).permute(0, 1, 4, 3, 2)
