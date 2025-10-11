############### TODO
# recursive hill climbing
# climb in an arbitrary direction (vector input)
###############
import math
import logging
import sigfig
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from datetime import datetime
from collections.abc import Sequence as sequence    # sorry
from collections.abc import Callable
from collections import deque
from typing import Optional, Union, Sequence, Tuple, List, Type
from numbers import Real

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
        step = -step

    step_sizes = []
    climber_positions = []
    for i, results in enumerate(climber_results):
        climb_positions = []
        for j, result in enumerate(results):
            climb_positions.append(position.microns)
            position += step
        position -= step    # do not move after stopping climb
        climber_positions.append(climb_positions)
        step_sizes.append(step)
        step = -1 * step * step_factor

    ncols = math.ceil(math.sqrt(len(climber_results)))
    nrows = math.ceil(len(climber_results) / ncols)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, layout='constrained',
                            figsize=(3*ncols, 0.2+3*nrows))
    if nrows == ncols == 1:
        axs = np.array([axs, ])

    for n, (ax, positions, results, step_size) in \
            enumerate(zip(axs.flatten(), climber_positions, climber_results, step_sizes)):
        start = sigfig.round(positions[0], step_size.microns,
                                   sep=' ', output_type=str).split()[0]
        startlabel = f"start ({start}um, {sigfig.round(results[0], 6)})"
        end = sigfig.round(positions[-1], step_size.microns,
                                   sep=' ', output_type=str).split()[0]
        endlabel = f"end ({end}um, {sigfig.round(results[-1], 6)})"

        ax.scatter(positions[:1], results[:1], c='g', marker='+',
                   label=startlabel, s=25, linewidth=1)
        ax.scatter(positions[1:-1], results[1:-1], c='b', marker='x',
                   s=15, linewidth=0.7, alpha=0.6,
                   label=f"step size =\n{step_size.prettyprint(stacked=True)}")
        ax.scatter(positions[-1:], results[-1:], c='r', marker='+',
                   label=endlabel, s=25, linewidth=1)

        ax.set_xlabel(f"{axis} (microns)")
        ax.set_title(str(n), fontsize=12)
        ax.legend(fontsize=6, framealpha=0.3)
        ax.tick_params(axis='both', which='both', labelsize=6)
        ax.grid(alpha=0.4, zorder=0)

    if len(climber_results) < len(axs.flatten()):
        for n in range(len(climber_results), len(axs.flatten())):
            axs.flatten()[n].axis('off')

    suptitle_fontsize = 7 + 4 * ncols
    if suptitle_fontsize > 15:
        suptitle_fontsize = 15
    fig.suptitle(f"{stagename} axis {axis} climbing with {movementType.value}",
                 fontsize=suptitle_fontsize)

    return fig


def hill_climb(stage: StageDevices, axis: str, movementType: MovementType,
               step: Distance, max_steps: int, softening: float,
               Ndecrease: int, exposureTime: Union[int, float]):

    log.info(f"Hill climbing on {stage.name} axis {axis} {movementType.value}" +
            f" with step size {step.prettyprint()} and exposure time {exposureTime}")

    step_size = copy(step)

    last = deque(maxlen=Ndecrease)
    current = stage.integrate(exposureTime)
    results = [current, ]
    for n in range(max_steps):
        last.append(current + abs(current) * softening)
        _ = stage.move(axis, step_size, movementType)
        current = stage.integrate(exposureTime)
        results.append(current)

        if len(last) == Ndecrease and current < sum(last) / Ndecrease:
            break
    else:
        log.warning(f"Climb hit max steps of {max_steps} without finding a peak")
        return np.array(results), False

    return np.array(results), True


def hill_climber(stage: StageDevices, axis: str, exposureTime: Union[int, float],
                 movementType: MovementType, init_step: Distance, step_factor: float,
                 init_positive: bool, rel_tol: float, softening: float,
                 abs_tol: Distance, Ndecrease: int, max_climbs: int,
                 max_steps: int, min_step: Distance, show_plot: bool = False, log_plot: bool = True,
                 recurse_args: Sequence[Sequence] = [], recurse_kwargs: Sequence[dict] = []):

    assert movementType != MovementType.GENERAL, "General movement hill climb may be added later"

    if init_positive is None:
        init_positive = True
        try_again = True
    else:
        try_again = False

    log.info(f"Activating hill climber on {stage.name} axis {axis} {movementType.value}" +
            f" initially in the {'positive' if init_positive else 'negative'} direction" +
             f" with exposure time {exposureTime} and initial step size {init_step.prettyprint()}")

    init_position = Distance(0, 'microns')
    if movementType == MovementType.PIEZO or movementType == MovementType.GENERAL:
        init_position += stage.axes[axis].get_piezo_position()
    if movementType == MovementType.STEPPER or movementType == MovementType.GENERAL:
        init_position += stage.axes[axis].get_stepper_position()

    step = copy(init_step)
    if not init_positive:
        step = -step

    last = -np.inf
    current = stage.integrate(exposureTime)
    climber_results = []
    n = 0
    while n < max_climbs:
        last = current
        if len(recurse_args) > 0:
            run(stage, *recurse_args, **recurse_kwargs)
        results, success = hill_climb(stage, axis, movementType, step,
                          max_steps, softening, Ndecrease, exposureTime)
        if not success:
            log.info("Hill climber stopped because last climb did not find peak")
            break
        current = results.max()
        climber_results.append(results)

        if current / last < 1 + rel_tol:
            # go back to highest recorded point
            stage.move(axis, step * (results.argmax() - len(results) + 1), movementType)
            log.info("Hill climber successfully converged to within" +
                     f" relative tolerance of {sigfig.round(rel_tol, 3, warn=False)}" +
                     f" at {axis} = {stage.axes[axis].get_stepper_position().prettyprint()}")
            if n == 0 and try_again:    # if init_positive was None, try both directions
                log.info("Trying again initially in the " +
                         f"{'positive' if init_positive else 'negative'} direction")
                step = -step
                try_again = False
                last = -np.inf
                climber_results = []
                continue
            break

        if current / last < abs_tol:
            # go back to highest recorded point
            stage.move(axis, step * (results.argmax() - len(results) + 1), movementType)
            log.info("Hill climber succesfully converged to within" +
                     f" absolute tolerance of {sigfig.round(abs_tol, 3, warn=False)}" +
                     f" at {axis} = {stage.axes[axis].get_stepper_position().prettyprint()}")
            if n == 0 and try_again:    # if init_positive was None, try both directions
                log.info("Trying again initially in the " +
                         f"{'positive' if init_positive else 'negative'} direction")
                step = -step
                try_again = False
                last = -np.inf
                climber_results = []
                continue
            break

        step = step * step_factor
        softening = softening * step_factor
        step = -step

        if abs(step) < min_step:
            log.warning(f"Hill climber hit minimum step size {min_step.prettyprint()}, " +
                        "consider incresing tolerances")
            break
        n += 1

    if n >= max_climbs:
        log.warning(f"Hill climber hit max climb limit of {max_climbs} withough converging")

    if show_plot or log_plot:
        fig = plot_climber(climber_results, stage.name, axis,
                           init_step, step_factor, init_position, init_positive, movementType)
        log.info("plot_generated")
        if log_plot:
            fig.savefig(f"./log_plots/{str(datetime.now())[:-7].replace(' ', '_')}_" +
                f"{stage.name}_{axis}_{movementType.value}_HillClimb.png",
                        format='png', facecolor='white', dpi=200)
            log.info("plot saved to ./log_plots")
        if show_plot:
            plt.show(block=True)
        plt.close()

    return


def arg_check(arg, argname, argtype: Type, axes,
              extra: Optional[Callable] = None, extra_text: str = ''):
    # for checking additional conditions
    if extra is None:
        extra = lambda x: True      # noqa E731

    if isinstance(arg, argtype):
        if extra(arg):
            return [arg, ] * len(axes)
        else:
            raise ValueError(f"{argname} ({arg}) must satisfy: {extra_text}")
    if isinstance(arg, sequence) and \
            all(isinstance(elem, argtype) and extra(elem) for elem in arg):
        if len(arg) == len(axes):
            return arg
        raise ValueError(f"{argname} ({arg}) must have same length as axes")
    raise ValueError(f"{argname} ({arg}) must be a {argtype.__name__} {extra_text} or a sequence thereof")


def run(stage: StageDevices,
        movementType: Union[MovementType, Sequence[MovementType]],
        exposureTime: Union[int, float, Sequence[Union[int, float]]],
        axes: Optional[str] = None,
        init_step: Union[None, Distance, Sequence[Distance]] = None,
        step_factor: Union[float, Sequence[float]] = 0.5,
        init_positive: Union[None, bool, Sequence[Union[None, bool]]] = None,
        rel_tol: Union[float, Sequence[float]] = 1e-2,
        softening: Union[float, Sequence[float]] = 0.0,
        abs_tol: Union[float, Sequence[float]] = 0.0,
        Ndecrease: Union[int, Sequence[int]] = 1,
        max_climbs: Union[int, Sequence[int]] = 16,
        max_steps: Union[int, Sequence[int]] = 100,
        min_step: Union[None, Distance, Sequence[Distance]] = None,
        order: Optional[Tuple[int, ...]] = None,
        show_plot: Union[bool, Sequence[bool]] = False,
        log_plot: Union[bool, Sequence[bool]] = True):
    # axes are the axes you want to hill climb on
    # movementType can be general, and must either be a single or have same length as axes
    # init_step is the initial step size either for all axes or for each axis
    # step_factor is multiplied by init_step when the peak is overshot
    # init_positive can tell whether to start moving forwards or backwards first
    # order determines whether to hill climb RECURSIVELY, not just which axis to do first

    # default values
    if axes is None:
        axes = 'xzy'
    else:
        axes = list(axes)

    if movementType is None:
        movementType = MovementType.GENERAL

    # has to go before the for loops below
    movementType = arg_check(movementType, 'movementType', MovementType, axes)

    if min_step is None:
        min_step = []
        for movetype in movementType:
            if movetype == MovementType.PIEZO or movetype == MovementType.GENERAL:
                min_step.append(Distance(0.1, 'volts'))
            if movetype == MovementType.STEPPER:
                min_step.append(Distance(1, 'steps'))

    if init_step is None:
        init_step = []
        for movetype in movementType:
            if movetype == MovementType.STEPPER or movetype == MovementType.GENERAL:
                init_step.append(Distance(100, 'microns'))
            if movetype == MovementType.PIEZO:
                init_step.append(Distance(1, 'microns'))

    if order is None:
        order = 0

    # veryfing and ducking inputs
    assert all(ax.lower() in VALID_AXES for ax in axes), "axes must be x, y, or z"
    assert isinstance(exposureTime, int) or isinstance(exposureTime, float)

    Gthan0 = (lambda x: x > 0, 'greater than zero')     # common requirements
    Geqthan0 = (lambda x: x >= 0, 'greater than or equal to zero')

    exposureTime = arg_check(exposureTime, 'exposureTime', (int, float), axes)
    init_step = arg_check(init_step, 'init_step', Distance, axes)
    step_factor = arg_check(step_factor, 'step_factor', Real, axes,
                            lambda f: 0 < f < 1, "between 0 and 1")
    init_positive = arg_check(init_positive, 'init_positive', (bool, type(None)), axes)
    rel_tol = arg_check(rel_tol, 'rel_tol', Real, axes, *Geqthan0)
    softening = arg_check(softening, 'softening', Real, axes, *Geqthan0)
    abs_tol = arg_check(abs_tol, 'abs_tol', Real, axes)
    Ndecrease = arg_check(Ndecrease, 'Ndecrease', int, axes, *Gthan0)
    max_climbs = arg_check(max_climbs, 'max_climbs', int, axes, *Gthan0)
    max_steps = arg_check(max_steps, 'max_steps', int, axes, *Gthan0)
    min_step = arg_check(min_step, 'min_step', Distance, axes)
    order = arg_check(order, 'order', int, axes)
    show_plot = arg_check(show_plot, 'show_plot', bool, axes)
    log_plot = arg_check(log_plot, 'log_plot', bool, axes)

    # could probably do this better using the inspect module
    run_args = np.array([movementType, exposureTime, axes], dtype=object)
    hill_climber_kwargs_values = np.array([init_step, step_factor, init_positive,
                                   rel_tol, softening, abs_tol, Ndecrease,
                                   max_climbs, max_steps, min_step,
                                   show_plot, log_plot, order], dtype=object)
    hill_climber_kwargs_keys = np.array(['init_step', 'step_factor', 'init_positive',
                                'rel_tol', 'softening', 'abs_tol', 'Ndecrease',
                                'max_climbs', 'max_steps', 'min_step',
                                'show_plot', 'log_plot', 'order'], dtype=object)

    # reorder arg and kwarg values based on order of order
    descending_indices = np.argsort(order)[::-1]
    run_args = run_args.T[descending_indices].T
    hill_climber_kwargs_values = hill_climber_kwargs_values.T[descending_indices].T

    # if applicable, move args and kwargs of to-be-recursed axes into the recurse_(kw)args
    recurse_args = []
    recurse_kwargs = []
    for i in range(len(order) - 1):
        if order[i] > order[i+1]:
            recurse_args.append(run_args[:, i:])
            recurse_kwargs.append(hill_climber_kwargs_values[:, i:])
            # remove to-be-recursed kwargs from outer loop
            hill_climber_kwargs_values = np.delete(hill_climber_kwargs_values, i, 1)
            hill_climber_kwargs_keys = np.delete(hill_climber_kwargs_keys, i, 0)
            break
        else:
            pass

    breakpoint()
    recurse_kwargs = {key: value for key, value in zip(hill_climber_kwargs_keys, recurse_kwargs)}

    # hill_climber doesn't take order as a kwargs, but does take recurse_(kw)args
    outer_arg_length = len(hill_climber_kwargs_values[0])
    recurse_args = [recurse_args for i in range(outer_arg_length)]
    recurse_kwargs = [recurse_kwargs for i in range(outer_arg_length)]
    hill_climber_kwargs_values = hill_climber_kwargs_values[:-1].tolist()
    hill_climber_kwargs_values.append(recurse_args)
    hill_climber_kwargs_values.append(recurse_kwargs)
    hill_climber_kwargs_keys = list(hill_climber_kwargs_keys[:-1]) + ['recurse_args'] + ['recurse_kwargs']

    # run hill climbers
    for i, (axis, movetype) in enumerate(zip(axes, movementType)):
        kwargs = {key : hill_climber_kwargs_values[j][i] for j, key in enumerate(hill_climber_kwargs_keys)}
        _ = hill_climber(stage, axis, exposureTime, movetype, **kwargs)

    return
