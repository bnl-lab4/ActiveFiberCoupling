############### TODO
# better process for softening (require n points monatonically decreasing?)
# after each climb, move stage to first position after crossing the peak
# absolute tolerance and/or softening?
# recursive hill climbing?
###############
import math
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections.abc import Sequence as sequence    # sorry
from typing import Optional, Union, Sequence, Tuple, Dict, List

from MovementClasses import Distance, MovementType, StageDevices

# unique logger name for this module
log = logging.getLogger(__name__)

VALID_AXES = {'x', 'y', 'z'}


def plot_climber(climber_results: List[np.ndarray], stagename: str, axis: str,
                 init_step: Distance, step_factor: float, init_position: Distance,
                 init_positive: bool, movementType: MovementType):
    position = init_position
    step = init_step
    if not init_positive:
        step = -1 * step

    step_sizes = []
    climber_positions = []
    for i, results in enumerate(climber_results):
        climb_positions = []
        for j, result in enumerate(results):
            climb_positions.append(position.microns)
            position += step
        climber_positions.append(climb_positions)
        step = -1 * step * step_factor
        step_sizes.append(step)

    ncols = math.ceil(math.sqrt(len(climber_results)))
    nrows = math.ceil(len(climber_results) / ncols)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, layout='constrained',
                            figsize=(3*ncols, 0.2+3*nrows))
    for n, (ax, positions, results, step_size) in \
            enumerate(zip(axs.flatten(), climber_positions, climber_results, step_sizes)):
        ax.scatter(positions[:1], results[:1], c='g', marker='+', label='start',
                   s=20, linewidth=1)
        ax.scatter(positions[1:-1], results[1:-1], c='b', marker='x',
                   s=15, linewidth=0.7, alpha=0.6,
                   label=f"step size =\n{step_size.prettyprint(stacked=True)}")
        ax.scatter(positions[-1:], results[-1:], c='r', marker='+', label='end',
                   s=20, linewidth=1)

        ax.set_xlabel(f"{axis} (microns)")
        ax.set_title(str(n), fontsize=12)
        ax.legend(fontsize=6, framealpha=0.3)
        ax.tick_params(axis='both', which='both', labelsize=6)

    if len(climber_results) < len(axs.flatten()):
        for n in range(len(climber_results), len(axs.flatten())):
            axs.flatten()[n].axis('off')

    suptitle_fontsize = 7 + 4 * ncols
    if suptitle_fontsize > 15:
        suptitle_fontsize = 15
    fig.suptitle(f"{stagename} axis {axis} climbing with {movementType.value}",
                 fontsize=suptitle_fontsize)

    return fig


def hill_climb(stage: StageDevices, axis: str, movementType: MovementType, step: Distance,
               max_steps: int, positive: bool, softening: float, exposureTime: Union[int, float]):

    log.info(f"Hill climbing on {stage.name} axis {axis} {movementType.value} in the" +
            f" {'positive' if positive else 'negative'} direction with exposure time" +
             f" {exposureTime} and step size {step.prettyprint()}")

    if not positive:
        step = -1 * step

    last = -np.inf
    current = stage.integrate(exposureTime)
    results = [current, ]
    for n in range(max_steps):
        last = current
        _ = stage.move(axis, step, movementType)
        current = stage.integrate(exposureTime)
        results.append(current)

        if current * (1 + softening) < last:
            break

    results = np.array(results)
    return results


def hill_climber(stage: StageDevices, axis: str,
                 movementType: MovementType, init_step: float, step_factor: float,
                 tolerance: float, softening: float, exposureTime: Union[int, float],
                 init_positive: Optional[bool],
                 max_climbs: int = 100, max_steps: int = 100, show_plot: bool = False,
                 log_plot: bool = True, recurse: Optional[Dict[str, dict]] = None):

    # recurse on the key axes with value kwargs, IN ORDER (may not work on Python <3.7)
    assert recurse is None, "Recursion may be added later"
    assert movementType != MovementType.GENERAL, "General movement hill climb may be added later"

    log.info(f"Activating hill climber on {stage.name} axis {axis} {movementType.value}" +
            f" initially in the {'positive' if init_positive else 'negative'} direction" +
             f" with exposure time {exposureTime} and initial step size {init_step.prettyprint()}")

    init_position = Distance(0, 'microns')
    if movementType == MovementType.PIEZO or movementType == MovementType.GENERAL:
        init_position += stage.axes[axis].get_piezo_position()
    if movementType == MovementType.STEPPER or movementType == MovementType.GENERAL:
        init_position += stage.axes[axis].get_stepper_position()

    positive = init_positive
    step = init_step
    soften = softening

    last = -np.inf
    current = stage.integrate(exposureTime)
    climber_results = []
    for n in range(max_climbs):
        last = current
        results = hill_climb(stage, axis, movementType, step, max_steps,
                             positive, soften, exposureTime)
        current = results.max()
        climber_results.append(results)

        if current < (1 + tolerance) * last:
            break
        step = step * step_factor
        soften = soften * step_factor
        positive = not positive

    if show_plot or log_plot:
        fig = plot_climber(climber_results, stage.name, axis,
                           init_step, step_factor, init_position, init_positive, movementType)
        log.info("plot_generated")
        if log_plot:
            fig.savefig(f"./log_plots/{str(datetime.now())[:-7].replace(' ', '_')}_" +
                f"{stage.name}_{movementType.value}_HillClimb.png",
                        format='png', facecolor='white', dpi=200)
            log.info("plot saved to ./log_plots")
        if show_plot:
            plt.show(block=True)

    return


def run(stage: StageDevices, movementType: Union[MovementType, Tuple[MovementType, ...]],
        exposureTime: Union[int, float], axes: Optional[str] = None,
        init_step: Union[None, Distance, Tuple[Distance, ...]] = None,
        step_factor: Union[float, Tuple[float, ...]] = 2e-1,
        tolerance: float = 1e-2, softening: float = 1e-2,
        init_positive: Union[bool, Sequence[bool]] = True,
        max_climbs: int = 16, max_steps: int = 100,
        order: Optional[Tuple[int, ...]] = None,
        show_plot: bool = False, log_plot: bool = True):
    # axes are the axes you want to hill climb on
    # movementType can be general, and must either be a single or have same length as axes
    # init_step is the initial step size either for all axes or for each axis
    # step_factor is multiplied by init_step when the peak is overshot
    # init_positive can tell whether to start moving forwards or backwards first
    # order determines whether to hill climb RECURSIVELY, not just which axis to do first

    if axes is None:
        axes = 'xzy'

    assert all([ax.lower() in VALID_AXES for ax in axes]), "axes must be x, y, or z"

    if isinstance(movementType, MovementType):
        movementType = (movementType, ) * len(axes)
    elif isinstance(movementType, sequence) and \
            all([isinstance(elem, MovementType) for elem in movementType]):
        pass
    else:
        raise ValueError("movementType must be an enum MovementType or a sequence thereof")

    if init_step is None:
        init_step = []
        for movetype in movementType:
            if movetype == MovementType.STEPPER or movetype == MovementType.GENERAL:
                init_step.append(Distance(100, 'microns'))
            if movetype == MovementType.PIEZO:
                init_step.append(Distance(1, 'microns'))

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

    if isinstance(tolerance, float):
        if tolerance > 0:
            tolerance = (tolerance, ) * len(axes)
        else:
            raise ValueError("tolerance must be greater than zero")
    elif isinstance(tolerance, sequence) and \
            all([isinstance(elem, float) and tolerance > 0 for elem in tolerance]):
        pass
    else:
        raise ValueError("tolerance must be a float greater than zero or a sequence thereof")

    if isinstance(softening, float):
        if softening > 0:
            softening = (softening, ) * len(axes)
        else:
            raise ValueError("softening must be greater than zero")
    elif isinstance(softening, sequence) and \
            all([isinstance(elem, float) and softening > 0 for elem in softening]):
        pass
    else:
        raise ValueError("softening must be a float greater than zero or a sequence thereof")

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

    assert isinstance(show_plot, bool)
    assert isinstance(log_plot, bool)

    # FOR NOW, NO RECURSION
    assert order is None, "No recursion for now"

    assert len(axes) == len(movementType) == len(init_step) == \
            len(step_factor) == len(tolerance) == len(init_positive), \
            "these args must all have the same length"

    for axis, movetype, step, factor, tol, soft, positive in zip(
            axes, movementType, init_step, step_factor, tolerance, softening, init_positive):
        hill_climber(stage, axis, movetype, step, factor, tol, soft, exposureTime,
                     positive, max_climbs, max_steps, show_plot, log_plot)

    return
