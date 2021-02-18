from typing import Iterable


def get_step_choice_grid(step):
    if isinstance(step, Iterable):
        grid = [[substep, substep.get_tuning_grid()] for substep in step]
    else:
        grid = [step, step.get_tuning_grid()]
    return grid
