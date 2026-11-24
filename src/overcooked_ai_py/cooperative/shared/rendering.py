from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .layouts import COOKER_PORTS, STATIONS


def render(env):
    import pygame
    from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
    state = env.state
    def sprite(value, position):
        if value is None:
            return None
        if value.kind in ("raw_main", "raw_side", "chopped_side", "plate"):
            name = {"raw_main": "onion", "raw_side": "tomato", "chopped_side": "tomato", "plate": "dish"}[value.kind]
            return SimpleNamespace(name=name, position=position)
        return SimpleNamespace(name="soup", position=position, ingredients=["onion"], is_idle=False, is_ready=True, _cooking_tick=env.config.cook_steps, cook_time=env.config.cook_steps)
    players = [SimpleNamespace(position=state.chefs[i].position, orientation=state.chefs[i].orientation, held_object=sprite(state.chefs[i].held, state.chefs[i].position)) for i in env.active]
    objects = {}
    for code, value in (("K", state.cutting.item), ("B", state.buffer), ("G", state.cooker.item)):
        if value is not None:
            objects[STATIONS[code]] = sprite(value, STATIONS[code])
    if state.cooker.item is not None:
        pot = objects[STATIONS["G"]]
        pot.name = "soup"
        pot.ingredients = ["onion"]
        pot.is_idle = False
        pot.is_ready = state.cooker.remaining == 0
        pot._cooking_tick = env.config.cook_steps - state.cooker.remaining
        pot.cook_time = env.config.cook_steps
    grid = [[{"M": "O", "L": "T", "K": "X", "G": "P", "A": "X", "B": "X"}.get(tile, tile) for tile in row] for row in env.terrain]
    class Visualizer(StateVisualizer):
        def _render_grid(self, surface, terrain):
            super()._render_grid(surface, terrain)
            for position in COOKER_PORTS:
                x, y = self._position_in_unscaled_pixels(position)
                pygame.draw.rect(surface, (172, 70, 210), (x + 1, y + 1, 13, 13), 1)
    colors = ("red", "blue", "green")
    image = Visualizer(tile_size=80, is_rendering_hud=False, player_colors=[colors[i] for i in env.active]).render_state(SimpleNamespace(players=players, objects=objects), grid)
    pygame.font.init()
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    chinese = font_path.exists()
    font = pygame.font.Font(str(font_path) if chinese else None, 16)
    small = pygame.font.Font(str(font_path) if chinese else None, 13)
    labels = {"M": "主料", "L": "配菜", "K": "切菜", "G": "出锅", "A": "装配", "B": "暂存", "D": "餐盘", "S": "送餐"} if chinese else {"M": "Main", "L": "Side", "K": "Chop", "G": "Cook", "A": "Assemble", "B": "Buffer", "D": "Plates", "S": "Serve"}
    for code, (x, y) in STATIONS.items():
        label = small.render(labels[code], True, (255, 255, 255), (38, 48, 63))
        image.blit(label, (x * 80 + 3, y * 80 + 3))
    table = state.assembly
    for j, (label, value) in enumerate((("M", table.main), ("L", table.side), ("D", table.plate))):
        color = (65, 185, 100) if value else (88, 94, 105)
        rectangle = (3 * 80 + 3 + 25 * j, 3 * 80 + 34, 23, 23)
        pygame.draw.rect(image, color, rectangle, border_radius=3)
        image.blit(small.render("F" if value is not None and value.kind == "meal" else label, True, (255, 255, 255)), (rectangle[0] + 5, rectangle[1] + 2))
    if state.cooker.item is not None:
        kind = env.order(state.cooker.item.order_id).kind
        text = ("双人" if kind == "joint" else "单人") if chinese else kind
        image.blit(small.render(text, True, (255, 255, 255), (112, 50, 150)), (3 * 80 + 3, 2 * 80 + 57))
    if state.cutting.item is not None:
        image.blit(small.render(str(state.cutting.progress) + "/" + str(env.config.chop_steps), True, (255, 255, 255), (40, 90, 50)), (6 * 80 + 3, 2 * 80 + 57))
    for i in env.active:
        x, y = state.chefs[i].position
        number = small.render(str(i + 1), True, (255, 255, 255), (35, 40, 48))
        image.blit(number, (x * 80 + 4, y * 80 + 61))
    canvas = pygame.Surface((560, 640))
    canvas.fill((242, 245, 250))
    canvas.blit(image, (0, 80))
    title = "三人共享厨房" if chinese else "Three-chef shared kitchen"
    canvas.blit(font.render(title, True, (32, 45, 65)), (12, 3))
    metrics = env.episode_metrics()
    ticks = f'{metrics["ticks"]}/{env.config.horizon}'
    score = metrics["team_score"]
    total, normal, joint = (metrics[key] for key in ("meals_completed", "normal_completed", "joint_completed"))
    if chinese:
        status = f"步数 {ticks}    团队得分 {score:g}    出餐 {total}（普通 {normal} / 协作 {joint}）"
    else:
        status = f"Step {ticks}    Team score {score:g}    Meals {total} (normal {normal} / joint {joint})"
    canvas.blit(small.render(status, True, (32, 45, 65)), (12, 25))
    attempts = metrics["joint_attempt_steps"]
    if attempts:
        sync = f'{metrics["joint_attempt_success_rate"]:.1%} ({metrics["joint_successes"]}/{attempts})'
    else:
        sync = "暂无尝试" if chinese else "no attempts"
    wait = metrics["joint_wait_steps"]
    sync_status = f"同步出锅成功率 {sync}    同步等待 {wait} 步" if chinese else f"Sync success {sync}    Sync waiting {wait} steps"
    canvas.blit(small.render(sync_status, True, (32, 45, 65)), (12, 43))
    legend = "M=主料  L=配菜  D=餐盘  F=成品；紫框为左右出锅位" if chinese else "M=main  L=side  D=plate  F=meal; purple=unload positions"
    canvas.blit(small.render(legend, True, (32, 45, 65)), (12, 61))
    return np.asarray(pygame.surfarray.array3d(canvas)).transpose(1, 0, 2).copy()
