import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Sequence as sequence    # sorry
from typing import Optional, Union, Sequence, Tuple, Dict

from MovementClasses import Distance, MovementType

VALID_AXES = {'x', 'y', 'z'}


def plot_climb(axis: str, positions: Sequence[Distance], heights: Sequence[float],
               init_step: Distance, step_factor: float):
    return


def hill_climb(axis: str, movementType: MovementType, step: Distance,
               positive: bool, exposureTime: Union[int, float]):
    return


def hill_climber(axis: str, movementType: MovementType, init_step: float, step_factor: float,
                 exposureTime: Union[int, float], init_positive: Optional[bool],
                 recurse: Optional[Dict[str : dict, ...]] = None,
                show_plots: bool = False, log_plots: bool = True):
    # recurse on the key axes with value kwargs, IN ORDER (may not work on Python <3.7)
    return


def run(axes: str, movementType: Union[MovementType, Tuple[MovementType, ...]],
        init_step: Union[Distance, Tuple[Distance, ...]],
        step_factor: Union[float, Tuple[float, ...]],
        exposureTime: Union[int, float], init_positive: Union[None, bool, Sequence[bool]] = None,
        order: Optional[Tuple[int, ...]] = None,
        show_plots: bool = False, log_plots: bool = True):
    # axes are the axes you want to hill climb on
    # movementType can be general, and must either be a single or have same length as axes
    # init_step is the initial step size either for all axes or for each axis
    # step_factor is multiplied by init_step when the peak is overshot
    # init_positive can tell whether to start moving forwards or backwards first
    # order determines whether to hill climb RECURSIVELY, not just which axis to do first

    assert all([ax.lower() in VALID_AXES for ax in axes]), "axes must be x, y, or z"

    if isinstance(movementType, MovementType):
        movementType = (movementType, ) * len(axes)
    elif isinstance(movementType, sequence) and \
            all([isinstance(elem, MovementType) for elem in movementType]):
        pass
    else:
        raise ValueError("movementType must be an enum MovementType or a sequence thereof")

    if isinstance(init_step, Distance):
        init_step = (init_step, ) * len(axes)
    elif isinstance(init_step, sequence) and \
            all([isinstance(elem, Distance) for elem in init_step]):
        pass
    else:
        raise ValueError("init_step must be a Distance class or a sequence thereof")

    if isinstance(step_factor, float):
        if 0 < step_factor < 1:
            step_factor = (step_factor, ) * len(axes)
        else:
            raise ValueError("step_factor must be between 0 and 1")
    elif isinstance(step_factor, sequence) and \
            all([isinstance(elem, float) and 0 < step_factor < 1 for elem in step_factor]):
        pass
    else:
        raise ValueError("step_factor must be a float between 0 and 1 or a sequence thereof")

    if order is None:
        pass
    elif isinstance(order, sequence) and \
            all([isinstance(elem, int) for elem in order]):
        pass
    else:
        raise ValueError("order must be a None or a tuple of ints")

    if init_positive is None:
        pass
    elif isinstance(init_positive, bool):
        init_positive = (init_positive, ) * len(axes)
    elif isinstance(init_positive, sequence) and \
            all([isinstance(elem, bool) for elem in init_positive]):
        pass
    else:
        raise ValueError("init_positive must be None, a bool, or a sequence of bools")

    assert isinstance(show_plots, bool)
    assert isinstance(log_plots, bool)

    return
