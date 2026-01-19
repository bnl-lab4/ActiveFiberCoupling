"""
Optimize fiber position by repeatedly hill climbing with finer and finer step size.
"""

############### TODO
# recursive hill climbing
# climb in an arbitrary direction (vector input)
###############
import math
from collections import deque
from collections.abc import Callable
from collections.abc import Sequence as sequence  # sorry
from copy import copy
from datetime import datetime
from numbers import Real
from typing import List, Optional, Sequence, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import sigfig
from matplotlib.figure import Figure

from logging_utils import get_logger
from movement_classes import Distance, MovementType, StageDevices

# unique logger name for this module
logger = get_logger(__name__)

VALID_AXES = {"x", "y", "z"}


def plot_climber(
    climber_results: List[np.ndarray],
    stagename: str,
    axis: str,
    init_step: Distance,
    step_factor: float,
    init_position: Distance,
    init_positive: bool,
    movement_type: MovementType,
) -> Figure:
    """
    Plot the results of a series of hill climbs from `hill_climber` for a
    single axix.

    For each separate hill climb, plots the position vs sensor reading for
    each point and generates a legend with the step size and the position
    of the first and last positions. Recalculates the fiber positions based
    on the shape of `climber_results` and other args.

    Parameters
    ----------
    climber_results : list of np.ndarrays
        A list containing the results array from each call to `hill_climb`.
    stagename : str
        Name of the stage; used in the plot suptitle.
    axis : {'x', 'y', 'z'}
        Name of the axis which was climbed; used in the plot suptitle.
    init_step : distance.Distance
        Initial step size of the first hill climb.
    step_factor : float
        Step factor used to adjust step size after each climb.
    init_position : distance.Distance
        Initial fiber position. Used to calculate the fiber position at
        each step.
    init_positive : bool
        Whether the first hill climb was in the positive (``True``) or
        negative (``False``) direction.
    movement_type : movement_classes.MovementType
        Which movement type was used; used in the plot suptitle.

    Returns
    -------
    matplotlib.figure.Figure

    See Also
    --------
    hill_climber : calls this if the plot is to be shown or logged

    Notes
    -----
    This calculates the fiber positions because `hill_climb` uses ``move``
    commands, not ``goto`` commands. The logic in this function must be
    kept up to date (as the inverse) with `hill_climb` and `hill_climber`.
    """
    position = init_position
    step = init_step
    if not init_positive:
        step = -step

    step_sizes = []
    climber_positions = []
    for results in climber_results:
        climb_positions = []
        for _ in results:
            climb_positions.append(position.microns)
            position += step
        position -= step  # do not move after stopping climb
        climber_positions.append(climb_positions)
        step_sizes.append(step)
        step = -1 * step * step_factor

    ncols = math.ceil(math.sqrt(len(climber_results)))
    nrows = math.ceil(len(climber_results) / ncols)

    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        layout="constrained",
        figsize=(3 * ncols, 0.2 + 3 * nrows),
    )
    if nrows == ncols == 1:
        axs = np.array(
            [
                axs,
            ]
        )

    for n, (ax, positions, results, step_size) in enumerate(
        zip(axs.flatten(), climber_positions, climber_results, step_sizes)
    ):
        start = sigfig.round(
            positions[0], step_size.microns, sep=" ", output_type=str
        ).split()[0]
        startlabel = f"start ({start}um, {sigfig.round(results[0], 6)})"
        end = sigfig.round(
            positions[-1], step_size.microns, sep=" ", output_type=str
        ).split()[0]
        endlabel = f"end ({end}um, {sigfig.round(results[-1], 6)})"

        ax.scatter(
            positions[:1],
            results[:1],
            c="g",
            marker="+",
            label=startlabel,
            s=20,
            linewidth=1,
        )
        ax.scatter(
            positions[1:-1],
            results[1:-1],
            c="b",
            marker="x",
            s=15,
            linewidth=0.7,
            alpha=0.6,
            label=f"step size =\n{step_size.prettyprint(stacked=True)}",
        )
        ax.scatter(
            positions[-1:],
            results[-1:],
            c="r",
            marker="+",
            label=endlabel,
            s=20,
            linewidth=1,
        )

        ax.set_xlabel(f"{axis} (microns)")
        ax.set_title(str(n), fontsize=12)
        ax.legend(fontsize=6, framealpha=0.3)
        ax.tick_params(axis="both", which="both", labelsize=6)
        ax.grid(alpha=0.4, zorder=0)

    if len(climber_results) < len(axs.flatten()):
        for n in range(len(climber_results), len(axs.flatten())):
            axs.flatten()[n].axis("off")

    suptitle_fontsize = 7 + 4 * ncols
    if suptitle_fontsize > 15:
        suptitle_fontsize = 15
    fig.suptitle(
        f"{stagename} axis {axis} climbing with {movement_type.value}",
        fontsize=suptitle_fontsize,
    )

    return fig


def hill_climb(
    stage: StageDevices,
    axis: str,
    movement_type: MovementType,
    step: Distance,
    max_steps: int,
    softening: float,
    Ndecrease: int,
    exposure_time: Union[int, float],
):
    """
    Hill climb along a single step size with a single step size.

    Repeatedly moves `axis` by `step` and reads sensor until the sensor
    reading is below the highest recording for `Ndecrease` steps. Both
    `Ndecrease` and `softening` are intended to allow for increased
    robustness against low SNR measurements.
    Parameters
    ----------
    stage : movement_classes.StageDevices
        The stage to hill climb with.
    axis : {'x', 'y', 'z'}
        Single character str denoting the axis to hill climb on.
    movement_type : movement_classes.MovementType
        Which movement type to move with. Only ``PIEZO`` and ``STEPPER``
        are supported; ``GENERAL`` may be added at a later date.
    step : distance.Distance
        Step size.
    max_steps : int
        Maximum number of steps to take before stopping.
    softening : float
        Multiplicative factor on previous sensor values for determining
        whether to halt. Intended to help combat sensor noise.
    Ndecrease : int
        How many readings below the previous maximum to require before
        halting. Intended to help combat sensor noise.
    exposure_time : int or float
        What exposure time to use for the sensor readings.

    Returns
    -------
    np.ndarray
        Array of sensor readings from each step.
    bool
        Whether the function halted before reaching `max_steps`.

    See Also
    --------
    hill_climber : repeatedly calls `hill_climb` with decreasing step size.

    Notes
    -----
    Currently, `Ndecrease` only determines how many consecutive steps have
    to be taken where without finding a new maximum. The readings below the
    maximum do not need to monotonically decrease, though. It would be
    preferable to have a more robust system to account for noise and
    require monotonically decreasing measurements to halt. As it is now,
    `softening` is too simple and not very effective.

    See the "Notes" section of `run` regarding future additions of
    recursive hiil climbing.
    """
    logger.info(
        f"Hill climbing on {stage.name} axis {axis} {movement_type.value}"
        + f" with step size {step.prettyprint()} and exposure time {exposure_time}"
    )

    step_size = copy(step)

    last = deque(maxlen=Ndecrease)
    current = stage.integrate(exposure_time)
    results = [
        current,
    ]
    for _ in range(max_steps):
        last.append(current + abs(current) * softening)
        _ = stage.move(axis, step_size, movement_type)
        current = stage.integrate(exposure_time)
        results.append(current)

        if len(last) == Ndecrease and current < sum(last) / Ndecrease:
            break
    else:
        logger.warning(f"Climb hit max steps of {max_steps} without finding a peak")
        return np.array(results), False

    return np.array(results), True


def hill_climber(
    stage: StageDevices,
    axis: str,
    exposure_time: Union[int, float],
    movement_type: MovementType,
    init_step: Distance,
    step_factor: float,
    init_positive: bool,
    rel_tol: float,
    softening: float,
    abs_tol: Distance,
    Ndecrease: int,
    max_climbs: int,
    max_steps: int,
    min_step: Distance,
    show_plot: bool = False,
    log_plot: bool = True,
) -> None:
    """
    Hill climb along an axis with successively smaller step sizes to find a
    local maximum.

    Repeatedly call `hill_climb`, each time decreasing the step size by
    `step_factor` and switching the direction, until one of the relative or
    absolute tolerances are met or the minimum step size or maximum number
    of climbs are reached.

    Parameters
    ----------
    stage : movement_classes.StageDevices
        The stage to hill climb on.
    axis : {'x', 'y', 'z'}
        Single character str denoting the axis to hill climb on.
    exposure_time : float or int
        The exposure time to use for reading the sensor at each step.
    movement_type : movement_classes.MovementType
        Which movement type to move with. Only ``PIEZO`` and ``STEPPER``
        are supported; ``GENERAL`` may be added at a later date.
    init_step : distance.Distance
        The initial step size to use for the first climb.
    step_factor : float
        The amount by which to decrease the step size for each subsequent climb.
    init_positive : bool
        Whether to initially climb in the positive (``True``) or negative
        (``False``) direction.
    rel_tol : float
        Maximum ratio between previous and current hill climb peak values
        for which to halt.
    softening : float
        Multiplicative factor on previous sensor values for determining
        whether to halt. Intended to help combat sensor noise.
    abs_tol : distance.Distance
        Maximum difference between previous and current hill climb peak
        values for which to halt.
    Ndecrease : int
        How many readings below the previous maximum to require before
        halting. Intended to help combat sensor noise.
    max_climbs : int
        Maximum number of times to call `hill_climb`.
    max_steps : int
        Maximum number of steps to take in a single climb. Passed to `hill_climb`.
    min_step : distance.Distance
        Minimum step size to use without halting.
    show_plot : bool, default=False
        Whether to show a plot summarizing all hill climbs once halted. The
        plot blocks the script from continuing until it is closed.
    log_plot : bool, default=True
        Whether to save a plot summarizing all hill climbs. Saved in "./log_plots/".

    See Also
    --------
    hill_climb : hill climbs once with a constant step size

    Notes
    -----
    See the "Notes" section for `hill_climb` for a discussion on
    `softening` and `Ndecrease`.

    See the "Notes" section of `run` regarding future additions of
    recursive hiil climbing.
    """
    assert (
        movement_type != MovementType.GENERAL
    ), "General movement hill climb may be added later"

    logger.info(
        f"Activating hill climber on {stage.name} axis {axis} {movement_type.value}"
        + f" initially in the {'positive' if init_positive else 'negative'} direction"
        + f" with exposure time {exposure_time} and initial step size {init_step.prettyprint()}"
    )

    init_position = Distance(0, "microns")
    if movement_type == MovementType.PIEZO or movement_type == MovementType.GENERAL:
        init_position += stage.axes[axis].get_piezo_position()
    if movement_type == MovementType.STEPPER or movement_type == MovementType.GENERAL:
        init_position += stage.axes[axis].get_stepper_position()

    step = copy(init_step)
    if not init_positive:
        step = -step

    last = -np.inf
    current = stage.integrate(exposure_time)
    climber_results = []
    for n in range(max_climbs):
        last = current
        results, success = hill_climb(
            stage,
            axis,
            movement_type,
            step,
            max_steps,
            softening,
            Ndecrease,
            exposure_time,
        )
        if not success:
            logger.info("Hill climber stopped because last climb did not find peak")
            break
        current = results.max()
        climber_results.append(results)

        if current / last < 1 + rel_tol:
            print(current / last, 1 + rel_tol)
            # go back to highest recorded point
            stage.move(
                axis, step * (results.argmax() - len(results) + 1), movement_type
            )
            logger.info(
                "Hill climber successfully converged to within"
                + f" relative tolerance of {sigfig.round(rel_tol, 3, warn=False)}"
                + f" at {axis} = {stage.axes[axis].get_stepper_position().prettyprint()}"
            )
            break

        if current / last < abs_tol:
            print(current / last, abs_tol)
            # go back to highest recorded point
            stage.move(
                axis, step * (results.argmax() - len(results) + 1), movement_type
            )
            logger.info(
                "Hill climber succesfully converged to within"
                + f" absolute tolerance of {sigfig.round(abs_tol, 3, warn=False)}"
                + f" at {axis} = {stage.axes[axis].get_stepper_position().prettyprint()}"
            )
            break

        step = step * step_factor
        # from when softening was additive
        # softening = softening * step_factor
        step = -step

        if abs(step) < min_step:
            logger.warning(
                f"Hill climber hit minimum step size {min_step.prettyprint()}, "
                + "consider incresing tolerances"
            )
            break
    else:
        logger.warning(
            f"Hill climber hit max climb limit of {max_climbs} withough converging"
        )

    if show_plot or log_plot:
        fig = plot_climber(
            climber_results,
            stage.name,
            axis,
            init_step,
            step_factor,
            init_position,
            init_positive,
            movement_type,
        )
        logger.info("plot_generated")
        if log_plot:
            fig.savefig(
                f"./log_plots/{str(datetime.now())[:-7].replace(' ', '_')}_"
                + f"{stage.name}_{axis}_{movement_type.value}_hillclimb.png",
                format="png",
                facecolor="white",
                dpi=200,
            )
            logger.info("plot saved to ./log_plots")
        if show_plot:
            plt.show(block=True)
        plt.close()

    return


def arg_check(
    arg,
    argname,
    argtype: Type,
    axes,
    extra: Optional[Callable] = None,
    extra_text: str = "",
):
    """
    Used to duck-type many of the arguments of `run`.

    Duck-types `arg` into a list of `argtype`s if `arg` is not already a
    list. Restrictions on the value(s) of `arg` can be supplied via
    `extra`.

    Parameters
    ----------
    arg : any
        The argument to check and duck type.
    argname : str
        The name of the arg, for a potential error message.
    argtype : Type
        The type that `arg` or its elements should be.
    axes : str or list thereof
        `arg` is duck-typed to be the same length as `axes`.
    extra : callable, optional
        A function to place restrictions on the value(s) of `arg`. Must
        accept `arg` and return a bool.
    extra_text : str, default=""
        Additional text for a potential error message corresponding to the
        logic of `extra`.

    Returns
    -------
    list of type `argtype`
        This will be the same length as `axes`.

    Raises
    ------
    ValueError
        Raised if:
            * `arg` is not `argtype` or a list thereof
            * `arg` is a list that is not the same length as `axes`
            * `arg` or its elements do not satisfy `extra`
    """
    if extra is None:
        extra = lambda x: True  # noqa E731

    if isinstance(arg, argtype):
        if extra(arg):
            return [
                arg,
            ] * len(axes)
        else:
            raise ValueError(f"{argname} ({arg}) must satisfy: {extra_text}")
    if isinstance(arg, sequence) and all(
        isinstance(elem, argtype) and extra(elem) for elem in arg
    ):
        if len(arg) == len(axes):
            return arg
        raise ValueError(f"{argname} ({arg}) must have same length as axes")
    raise ValueError(
        f"{argname} ({arg}) must be a {argtype.__name__} {extra_text} or a sequence thereof"
    )


def run(
    stage: StageDevices,
    movement_type: Union[MovementType, Sequence[MovementType]],
    exposure_time: Union[int, float, Sequence[Union[int, float]]],
    axes: Union[None, str, List[str]] = None,
    init_step: Union[None, Distance, Sequence[Distance]] = None,
    step_factor: Union[float, Sequence[float]] = 0.5,
    init_positive: Union[bool, Sequence[bool]] = True,
    rel_tol: Union[float, Sequence[float]] = 1e-2,
    softening: Union[float, Sequence[float]] = 0.0,
    abs_tol: Union[float, Sequence[float]] = 0.0,
    Ndecrease: Union[int, Sequence[int]] = 1,
    max_climbs: Union[int, Sequence[int]] = 16,
    max_steps: Union[int, Sequence[int]] = 100,
    min_step: Union[None, Distance, Sequence[Distance]] = None,
    order: Optional[Tuple[int, ...]] = None,
    show_plot: Union[bool, Sequence[bool]] = False,
    log_plot: Union[bool, Sequence[bool]] = True,
) -> None:
    """
    Hill climb along one or more axes (individually) with successively smaller step sizes.

    Calls `hill_climber` on one or more axes. Almost all args can be given
    a single value to apply to all axes or a list of values, one for each axis.

    Parameters
    ----------
    stage : movement_classes.StageDevices
        The stage to hill climb on.
    movement_type : movement_classes.MovementType or list thereof
        Which movement type(s) to move with. Only ``PIEZO`` and ``STEPPER``
        are supported; ``GENERAL`` may be added at a later date.
    exposure_time : float, int, or list thereof
        The exposure time(s) to use for reading the sensor at each step.
    axes : str or list thereof, default=None
        String or list of axis names (``x``, ``y``, or ``z``) to hill climb
        on, in order. ``None`` defaults to ``"xzy"``.
    init_step : distance.Distance or list thereof, optional
        The initial step size(s) to use for the first climb. ``None``
        defaults to 1 or 100 microns for movement types ``PIEZO`` or
        ``STEPPER``, respectively.
    step_factor : float or list thereof, default=0.5
        The amount(s) by which to decrease the step size for each subsequent climb.
    init_positive : bool or list thereof, default=True
        Whether to initially climb in the positive (``True``) or negative
        (``False``) direction(s).
    rel_tol : float or list thereof, default=0.01
        Maximum ratio(s) between previous and current hill climb peak values
        for which to halt.
    softening : float or list thereof, default=0
        Multiplicative factor(s) on previous sensor values for determining
        whether to halt. Intended to help combat sensor noise.
    abs_tol : distance.Distance or list thereof, default=0
        Maximum difference(s) between previous and current hill climb peak
        values for which to halt.
    Ndecrease : int or list thereof, default=1
        How many readings below the previous maximum to require before
        halting. Intended to help combat sensor noise.
    max_climbs : int or list thereof, default=16
        Maximum number of times to call `hill_climb`.
    max_steps : int or list thereof, default=100
        Maximum number of steps to take in a single climb. Passed to `hill_climb`.
    min_step : distance.Distance or list thereof, default=None
        Minimum step size(s) to use without halting. ``None`` defaults to 0.1 volts or 1 microstep for movemente types ``PIEZO`` or ``STEPPER``, respectively.
    order : ``None``
        Reserved for future use with recursion.
    show_plot : bool or list thereof, default=False
        Whether to show a plot(s) summarizing all hill climbs once halted. The
        plot(s) blocks the script from continuing until they are closed.
    log_plot : bool or list thereof, default=True
        Whether to save a plot(s) summarizing all hill climbs. Saved in "./log_plots/".

    See Also
    --------
    arg_check : duck types each argument as needed
    hill_climber : called on each individual axis

    Notes
    -----
    See the "Notes" section for `hill_climb` for a discussion on
    `softening` and `Ndecrease`.

    In the future, this module will be updated to allow for recursive hill
    climbing, in which `run` is called on one or more axes between each
    step-movement and sensor reading in `hill_climb` for another axis. This
    could be used to hill climb in the focus direction while hill climbing
    in the transverse directions after each step to remain centered on the
    beam, for example.
    """
    # axes are the axes you want to hill climb on
    # movement_type can be general, and must either be a single or have same length as axes
    # init_step is the initial step size either for all axes or for each axis
    # step_factor is multiplied by init_step when the peak is overshot
    # init_positive can tell whether to start moving forwards or backwards first
    # order determines whether to hill climb RECURSIVELY, not just which axis to do first

    # FOR NOW, NO RECURSION
    assert order is None, "No recursion for now"

    # default values
    if axes is None:
        axes = "xzy"

    if movement_type is None:
        movement_type = MovementType.GENERAL

    # has to go before the for loops below
    movement_type = arg_check(movement_type, "movement_type", MovementType, axes)

    if min_step is None:
        min_step = []
        for movetype in movement_type:
            if movetype == MovementType.PIEZO or movetype == MovementType.GENERAL:
                min_step.append(Distance(0.1, "volts"))
            if movetype == MovementType.STEPPER:
                min_step.append(Distance(1, "steps"))

    if init_step is None:
        init_step = []
        for movetype in movement_type:
            if movetype == MovementType.STEPPER or movetype == MovementType.GENERAL:
                init_step.append(Distance(100, "microns"))
            if movetype == MovementType.PIEZO:
                init_step.append(Distance(1, "microns"))

    if order is None:
        order = 0

    # veryfing and ducking inputs
    assert all(ax.lower() in VALID_AXES for ax in axes), "axes must be x, y, or z"
    assert isinstance(exposure_time, int) or isinstance(exposure_time, float)

    Gthan0 = (lambda x: x > 0, "greater than zero")  # common requirements
    Geqthan0 = (lambda x: x >= 0, "greater than or equal to zero")

    init_step = arg_check(init_step, "init_step", Distance, axes)
    step_factor = arg_check(
        step_factor, "step_factor", Real, axes, lambda f: 0 < f < 1, "between 0 and 1"
    )
    init_positive = arg_check(init_positive, "init_positive", bool, axes)
    rel_tol = arg_check(rel_tol, "rel_tol", Real, axes, *Geqthan0)
    softening = arg_check(softening, "softening", Real, axes, *Geqthan0)
    abs_tol = arg_check(abs_tol, "abs_tol", Real, axes)
    Ndecrease = arg_check(Ndecrease, "Ndecrease", int, axes, *Gthan0)
    max_climbs = arg_check(max_climbs, "max_climbs", int, axes, *Gthan0)
    max_steps = arg_check(max_steps, "max_steps", int, axes, *Gthan0)
    min_step = arg_check(min_step, "min_step", Distance, axes)
    order = arg_check(order, "order", int, axes)
    show_plot = arg_check(show_plot, "show_plot", bool, axes)
    log_plot = arg_check(log_plot, "log_plot", bool, axes)

    # not the best way of doing this
    hill_climber_kwargs_values = [
        init_step,
        step_factor,
        init_positive,
        rel_tol,
        softening,
        abs_tol,
        Ndecrease,
        max_climbs,
        max_steps,
        min_step,
        show_plot,
        log_plot,
    ]
    hill_climber_kwargs_keys = [
        "init_step",
        "step_factor",
        "init_positive",
        "rel_tol",
        "softening",
        "abs_tol",
        "Ndecrease",
        "max_climbs",
        "max_steps",
        "min_step",
        "show_plot",
        "log_plot",
    ]
    hill_climber_kwargs = zip(hill_climber_kwargs_keys, hill_climber_kwargs_values)

    # run hill climbers
    for i, (axis, movetype) in enumerate(zip(axes, movement_type)):
        kwargs = {key: val[i] for key, val in hill_climber_kwargs}
        _ = hill_climber(stage, axis, exposure_time, movetype, **kwargs)

    return
