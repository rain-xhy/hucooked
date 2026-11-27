"""Physical-step reward terms used by the Ours training environment."""

from overcooked_ai_py.cooperative.shared import INTERACT
from overcooked_ai_py.cooperative.shared.layouts import STATIONS


STAGE_VALUES = {
    'held_raw_main': .1, 'held_raw_side': .1, 'held_plate': .1,
    'loaded_main': .8, 'cooked_main': 1.2, 'main_at_assembly': 2.,
    'side_on_board': .4, 'fully_cut_side_on_board': 1.,
    'held_chopped_side': 1.2, 'side_at_assembly': 2.,
    'plate_at_assembly': .5, 'meal_at_assembly': 6., 'held_meal': 6.5,
}

def potential_parts(core):
    state = core.state
    values = STAGE_VALUES
    held = sum(0. if chef.held is None else values.get('held_' + chef.held.kind, 0.)
               for chef in state.chefs)
    cooker = 0.
    if state.cooker.item is not None:
        fraction = 1. - state.cooker.remaining / core.config.cook_steps
        cooker = values['loaded_main'] + (values['cooked_main'] - values['loaded_main']) * fraction
    cutting = 0.
    if state.cutting.item is not None:
        fraction = state.cutting.progress / core.config.chop_steps
        cutting = values['side_on_board'] + (values['fully_cut_side_on_board'] - values['side_on_board']) * fraction
    assembly = state.assembly
    if assembly.plate is not None and assembly.plate.kind == 'meal':
        assembled = values['meal_at_assembly']
    else:
        assembled = (values['main_at_assembly'] if assembly.main is not None else 0.)
        assembled += values['side_at_assembly'] if assembly.side is not None else 0.
        assembled += values['plate_at_assembly'] if assembly.plate is not None else 0.
    buffer_values = {'raw_main': .1, 'raw_side': .1, 'plate': .1, 'chopped_side': 1., 'meal': 6.}
    buffer = 0. if state.buffer is None else buffer_values.get(state.buffer.kind, 0.)
    return dict(held=held, cooker=cooker, cutting=cutting, assembly=assembled, buffer=buffer)

def potential(core):
    return float(sum(potential_parts(core).values()))


KEY_BONUSES = {'load': 2., 'unload': 3., 'chop_complete': 2., 'assemble': 5.}

def milestone_counts(before, after, events):
    keys = set()
    for event in events:
        kind = event['type']
        if kind in ('load', 'unload', 'assemble'):
            order_id = event['order_id']
            field = {'load': 'started_at', 'unload': 'unloaded_at', 'assemble': 'assembled_at'}[kind]
            if getattr(before.orders[order_id], field) < 0 <= getattr(after.orders[order_id], field):
                keys.add((kind, order_id))
        elif kind == 'chop' and before.cutting.item is not None and after.cutting.item is not None:
            first = before.cutting.item
            last = after.cutting.item
            if first.kind == 'raw_side' and last.kind == 'chopped_side' and first.units == last.units:
                keys.add(('chop_complete', tuple(first.units)))
    return {kind: sum(key[0] == kind for key in keys) for kind in KEY_BONUSES}, sorted(keys, key=str)

def reward_parts(before, after, events, physical_actions, context, external, phi_before, phi_after):
    counts, keys = milestone_counts(before, after, events)
    returns = sum(event['type'] == 'return_source' for event in events)
    wrong = sum(action == INTERACT and held not in (None, 'meal') and front == STATIONS['S']
                for action, (held, front) in zip(physical_actions, context))
    parts = dict(external_raw=float(external), serving_base=10. * float(external),
                 progress=10. * (.99 * phi_after - phi_before),
                 key_steps=sum(KEY_BONUSES[kind] * count for kind, count in counts.items()),
                 time=-.1, return_source_penalty=-2. * returns, wrong_delivery_penalty=-5. * wrong,
                 return_source_count=returns, wrong_delivery_count=wrong)
    parts.update({kind + '_count': count for kind, count in counts.items()})
    return parts, keys


def team_reward(parts):
    return sum(parts[key] for key in ('serving_base', 'progress', 'key_steps', 'time',
                                     'return_source_penalty', 'wrong_delivery_penalty'))
