"""Grid search and fitting in one or more planes"""

########### TODO:
# disambiguate beam width vs waist
# deviation angles in 3d fit
###########

import copy
from collections.abc import Sequence as sequence
from datetime import datetime
from typing import List, Optional, Sequence, SupportsFloat, Union, cast

import lmfit
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import sigfig

from distance import Distance
from grid_plotting import (
    plot_2dfit,
    plot_3dfit,
    plot_lin_fit,
    plot_para_fit,
    plot_plane,
)
from logging_utils import get_logger
from movement_classes import MovementType, StageDevices

# unique logger name for this module
logger = get_logger(__name__)

# max numpy arr size before it summarizes when printed
np.set_printoptions(threshold=2000, linewidth=250)

VALID_AXES = {"x", "y", "z"}
WAVELENGTH = 0.65  # microns


def accept_fit():
    """
    Simple input loop about whether to accept or reject a fit.

    Returns
    -------
    bool
        Whether the fit is accepted (``True``) or rejected (``False``).
    """
    while True:
        print("Do you accept the fit?")
        user_input = input("(y/n): ").strip().lower()

        if user_input == "n":
            logger.info("Fit rejected")
            return False
        if user_input == "y":
            logger.info("Fit accepted")
            return True

        print("Could not interpret; input must be 'y' or 'n'.")


def gaussbeam(
    x1: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
    waistx1: float = 0.0,
    waistx2: float = 0.0,
    waistx3: float = 0.0,
    I0: float = 1.0,
    w0: float = 8 * np.pi * WAVELENGTH,
    C: float = 0.0,
):
    r"""
    Model of a Gaussian beam.

    The propagation direction of the Gaussian beam is along the `x3` axis.

    Parameters
    ----------
    x1, x2, x3 : np.ndarray
        The values at which to return the beam intensity.
    waistx1, waistx2, waistx3 : float, default=0.0
        Position of the beam waist along the corresponding axes.
    I0 : float, default=1.0
        Peak intensity of the beam at the waist.
    w0 : float, default=8*pi*``WAVELENGTH``
        Waist size, e^2 full width
    C : float, default=0.0
        Constant background level.

    Returns
    -------
    np.ndarray
        Beam power at the positions defined by `x1`, `x2`, `x3`.

    Notes
    -----
    A Gaussian beam is one that has a transverse intensity profile of a
    2-dimensional Gaussian whose width changes along the propagation axis.
    A Gaussian beam is defined by its peak intensity $I_0$, waist $w_0$,
    wavelength $\lambda$, waist position $\vec{r}_0 = (x1_0, x2_0, x3_0)$,
    and propagation direction $\vec{k} = (k_1, k_2, k_3)$. To find the the
    intensity at position $\vec{r}$, it is convinient to define
    $r_\parallel = (\vec{r} - \vec{r}_0) \cdot \vec{k}$ and $r_\perp =
    \sqrt{\|\vec{r} - \vec{r}_0 \|^2 - r_\parallel^2}$, which are the
    distances from $\vec{r}$ to $\vec{r}_0$ parallel and perpendicular to
    $\vec{k}$, respectively. Then, the beam intensity is given by
    $$I(\vec{r}) = I_0 \left(\frac{w_0}{w(r_\parallel)} \right(^2 \exp \left(
    \frac{-2 r_\perp^2}{w(r_\parallel)^2} \right)$$ where $w(r_\parallel) =
    w_0 \sqrt{ 1 + \left( r_\parallel/r_R \right)^2}$ is the $1/e^2$
    diameter of the spot and $r_R = \pi w_0^2 n/ \lambda$ is the Rayleigh
    range, with $n$ being the refractive index of the medium. For
    convenience, the beam waist $w_0$ is calculated from the provided focal
    ratio $f$. $$w_0 = \frac{\pi \lambda}{\sin \left( \arctan
    \left( 1 / 2f \right) \right)}$$ The total power contained the beam is
    $P = \frac{1}{2} \pi w_0^2 I_0$.
    """
    pR2 = (
        np.pi * w0**2 / 0.65
    ) ** 2  # squared Rayleigh range in the propagation direction

    r2 = (x1 - waistx1) ** 2 + (
        x2 - waistx2
    ) ** 2  # squared distance from propagation axis
    p2 = (x3 - waistx3) ** 2  # squared position along propagation axis
    wp2 = w0**2 * (1 + p2 / pR2)  # squared waist at the given x3 position

    return I0 * (w0**2 / wp2) * np.exp(-2 * r2 / wp2) + C


def gaussbeam_fit_3d(
    axes: Union[str, List[str]],
    movement_type: MovementType,
    stagename: str,
    axis0_cube: np.ndarray,
    axis1_cube: np.ndarray,
    focus_cube: np.ndarray,
    data_cube: np.ndarray,
    show_plot: bool = False,
    log_plot: bool = True,
):
    """
    Fits 3D intensity data to a single Gaussian beam model.

    Performs the fit with the lmfit package, using `gaussbeam` as the model
    function.

    Parameters
    ----------
    axes : str, list of str
        Axis labels for the two non-focus axes. Must have length 2, where
        the 0th and 1st names correspond to `axis0_cube` and `axis1_cube`,
        respectively. Used for plotting.
    movement_type : `movement_classes.MovementType`
        What movement type was used to move around the individual planes
        (may be different from the movement type along the focus axis).
        Used for plotting.
    stagename : str
        Name of the stage. Used for plotting.
    axis0_cube, axis1_cube : np.ndarray
        Positions of the data points along the two axes given in `axes`
        (the two non-focus axes).
    focus_cube : np.ndarray
        Positions of the data points along the focus axis.
    data_cube : np.ndarray
        Recorded intensities at the points given by `axis0_cube`,
        `axis1_cube`, and `focus_cube`.
    show_plot : bool, default=False
        Whether to display a plot of the fit result. Opens a new window
        containing the plot which must be closed for the program to
        continue.
    log_plot : bool, default=True
        Whether to save the plot of the fit result. The plot's filepath
        will be
        ``./log_plots/yyyy-mm-dd_hh:mm:ss_<stagename>_<movement_type>_Gaussian_beam_fit.png``.

    Notes
    -----
    This currently assumes that the propagation direction is exactly along
    the "focus axis". In actuality, these are not perfectly aligned,
    causing the fit to not perform well. In the future, this may be updated
    to account for this misalignment via the addition of two additional
    "propagation-angle" free parameters.
    """
    assert (
        axis0_cube.shape == axis1_cube.shape == focus_cube.shape == data_cube.shape
    ), "All position/data inputs must have the same shape"

    model = lmfit.Model(gaussbeam, independent_vars=["x1", "x2", "x3"])
    params = model.make_params()

    # guess initial values
    params["I0"].value = data_cube.max()
    max_idx = np.unravel_index(data_cube.argmax(), data_cube.shape)
    params["waistx1"].value = axis0_cube[max_idx]
    params["waistx2"].value = axis1_cube[max_idx]
    params["waistx3"].value = focus_cube[max_idx]

    weights = data_cube
    weights = weights - weights.min() + 1e-6 * np.ones_like(weights)
    weights /= weights.max()
    weights = np.sqrt(weights)

    result = model.fit(
        data=data_cube.flatten(),
        x1=axis0_cube.flatten(),
        x2=axis1_cube.flatten(),
        x3=focus_cube.flatten(),
        params=params,
        weights=weights.flatten(),
    )
    logger.debug("Fit complete")

    # plot logic
    if log_plot or show_plot:
        fig = plot_3dfit(
            axes,
            axis0_cube,
            axis1_cube,
            focus_cube,
            result,
        )
        logger.info("Plot Generated")
        if log_plot:
            fig.savefig(
                f"./log_plots/{str(datetime.now())[:-7].replace(' ', '_')}_"
                + f"{stagename}_{movement_type.value}_Gaussian_beam_fit.png",
                format="png",
                facecolor="white",
                dpi=200,
            )
            logger.info("Plot saved to ./log_plots")
        if show_plot:
            plt.show(block=True)
        plt.close()

    return result


def gaussbeam_fit_2d(
    axes: Union[str, List[str]],
    movement_type: MovementType,
    stagename: str,
    axis0_grid: np.ndarray,
    axis1_grid: np.ndarray,
    grid_values: np.ndarray,
    plane: Distance,
    show_plot: bool = False,
    log_plot: bool = True,
):
    """
    Fits 2D intensity data to a Gaussian profile.

    Performs the fit with the lmfit package, using the built-in model
    ``Gaussian2dModel``. The model is contrained to fit a circular Gaussian
    by setting "sigmay" equal to "sigmax".

    Parameters
    ----------
    axes : str, list of str
        Axis labels for the two non-focus axes. Must have length 2, where
        the 0th and 1st names correspond to `axis0_grid` and `axis1_grid`,
        respectively. Used to infer the focus axis and for plotting.
    movement_type : `movement_classes.MovementType`
        What movement type was used to move around the individual planes
        (may be different from the movement type along the focus axis).
        Used for plotting.
    stagename : str
        Name of the stage. Used for plotting.
    axis0_grid, axis1_grid : np.ndarray
        Positions of the data points along the two axes given in `axes`
        (the two non-focus axes).
    grid_values : np.ndarray
        Recorded intensities at the points given by `axis0_grid`,
        `axis1_grid`, and `plane`.
    plane : `Distance.Distance`
        Position of the 2d grid of values along the focus axis.
    show_plot : bool, default=False
        Whether to display a plot of the fit result. Opens a new window
        containing the plot which must be closed for the program to
        continue.
    log_plot : bool, default=True
        Whether to save the plot of the fit result. The plot's filepath
        will be
        ``./log_plots/yyyy-mm-dd_hh:mm:ss_2dfit_<stagename>_<movement_type>_<focus_axis>-{plane in microns}um.png``.
    """
    focus_axis = list(VALID_AXES.difference(set(axes)))[0]

    gmodel = lmfit.models.Gaussian2dModel()
    params = gmodel.guess(
        grid_values.flatten(), axis0_grid.flatten(), axis1_grid.flatten()
    )
    model = gmodel + lmfit.models.ConstantModel()
    params.add("c", value=grid_values.mean())
    params["sigmay"].exp = "sigmax"  # circular gaussian

    weights = grid_values
    weights = weights - weights.min() + 1e-6 * np.ones_like(weights)
    weights /= weights.max()
    weights = np.sqrt(weights)

    result = model.fit(
        grid_values.flatten(),
        x=axis0_grid.flatten(),
        y=axis1_grid.flatten(),
        params=params,
        weights=weights.flatten(),
    )
    # logic
    if log_plot or show_plot:
        fig = plot_2dfit(grid_values, axis0_grid, axis1_grid, axes, plane, result)
        logger.info(f"Plot Generated for plane {plane.prettyprint()}")
        if log_plot:
            fig.savefig(
                f"./log_plots/{str(datetime.now())[:-7].replace(' ', '_')}_2dfit_"
                + f"{stagename}_{movement_type.value}_{focus_axis}-{plane.microns}um.png",
                format="png",
                facecolor="white",
                dpi=200,
            )
            logger.info("Plot saved to ./log_plots")
        if show_plot:
            plt.show(block=True)
        plt.close()

    return result


def width_parafit(
    axes: Union[str, List[str]],
    movement_type: MovementType,
    stagename: str,
    planes: Sequence[Distance],
    widths: np.ndarray,
    widths_unc: np.ndarray,
    show_plot: bool = False,
    log_plot: bool = True,
):
    r"""
    Fits a parabola to beam width vs focus-axis distance data.

    The beam widths are expected to come from the results of
    `gaussbeam_fit_2d` for multiple planes (must have at least three).
    Performs the fit with lmfit the lmfit package, using the built-in
    quadratic model.

    Parameters
    ----------
    axes : str, list of str
        Axis labels for the two non-focus axes, from which the focus axis
        is inferred. Must have length 2. Also used for plotting.
    movement_type : `movement_classes.MovementType`
        What movement type was used to move along the focus axis. Used for
        plotting.
    stagename : str
        Name of the stage. Used for plotting.
    planes : sequence of `Distance.Distance` objects
        Positions along the focus axis for which the beam width was
        measured.
    widths : np.ndarray
        Beam widths as measured at the positions given in `planes`.
    widths_unc : np.ndarray
        Uncertainties on the beam widths.
    show_plot : bool, default=False
        Whether to display a plot of the fit result. Opens a new window
        containing the plot which must be closed for the program to
        continue.
    log_plot : bool, default=True
        Whether to save the plot of the fit result. The plot's filepath
        will be
        ``./log_plots/yyyy-mm-dd_hh:mm:ss_parafit_<stagename>_<movement_type>_w-vs-<focus_axis>.png``.

    Notes
    -----
    Throughout, "beam width" usually refers to the $\sigma$ parameter in a
    circular 2d Gaussian fit (see `gaussbeam_fit_2d`). For a gaussian beam,
    the beam width is proportional to the square of the distance from the
    beam waist along in the direction of propagation.
    """
    focus_axis = list(VALID_AXES.difference(set(axes)))[0]
    planes_microns = np.array([plane.microns for plane in planes])

    model = lmfit.models.QuadraticModel()
    params = model.guess(widths, x=planes_microns)
    para_result = model.fit(
        widths, x=planes_microns, params=params, weights=1 / widths_unc
    )

    if show_plot or log_plot:
        fig = plot_para_fit(axes, widths, widths_unc, planes_microns, para_result)
        logger.info("Plot Generated")
        if log_plot:
            fig.savefig(
                f"./log_plots/{str(datetime.now())[:-7].replace(' ', '_')}_parafit_"
                + f"{stagename}_{movement_type.value}_w-vs-{focus_axis}.png",
                format="png",
                facecolor="white",
                dpi=200,
            )
        if show_plot:
            plt.show(block=True)
        plt.close()

    return para_result


def peaks_linfit(
    axes: Union[str, List[str]],
    first_axis: bool,
    movement_type: MovementType,
    stagename: str,
    planes: Sequence[Distance],
    peak_pos: np.ndarray,
    peak_unc: np.ndarray,
    show_plot: bool = False,
    log_plot: bool = True,
):
    """
    Fits a line to beam waist position vs focus-axis position data.

    The waist positions are expected to come from the results of
    `gaussbeam_fit_2d` for multiple planes (must have at least three).
    Performs a linear fit with the lmfit package using the built-in linear
    model.

    Parameters
    ----------
    axes : str, list of str
        Axis labels for the two non-focus axes, from which the focus axis
        is inferred. Must have length 2. Also used for plotting.
    first_axis : bool
        Whether the data in `peak_pos` and `peak_unc` is for the first axis
        in `axes` (``True``) or the second (``False``).
    movement_type : `movement_classes.MovementType`
        What movement type was used to move around the individual planes
        (may be different from the movement type along the focus axis).
    stagename : str
        Name of the stage. Used for plotting.
    planes : sequence of `Distance.Distance` objects
        Positions along the focus axis for which the beam width was
        measured.
    peak_pos, peak_unc : np.ndarray
        The beam peak positions and uncertainties thereof for each of the
        focus axis positions given in `planes`.
    show_plot : bool, default=False
        Whether to display a plot of the fit result. Opens a new window
        containing the plot which must be closed for the program to
        continue.
    log_plot : bool, default=True
        Whether to save the plot of the fit result. The plot's filepath
        will be
        ``./log_plots/yyyy-mm-dd_hh:mm:ss_linfit_<stagename>_<movement_type>_<axis>-vs-<focus_axis>.png``,
        where <axis> is the label of the axis of interest, as determined by
        `axes` and `first_axis`.

    Notes
    -----
    If the beam propagation axis is not exactly colinear with the stage's
    "focus axis", then the position of the beam waist along the non-focus
    axes is dependent on the position of the waist along the focus axis.
    """
    focus_axis = list(VALID_AXES.difference(set(axes)))[0]
    planes_microns = np.array([plane.microns for plane in planes])

    model = lmfit.models.LinearModel()
    params = model.guess(peak_pos, x=planes_microns)
    lin_result = model.fit(
        peak_pos, x=planes_microns, params=params, weights=1 / peak_unc
    )

    if show_plot or log_plot:
        axis = axes[0] if first_axis else axes[1]
        fig = plot_lin_fit(
            axis, focus_axis, peak_pos, peak_unc, planes_microns, lin_result
        )
        logger.info("Plot Generated")
        if log_plot:
            fig.savefig(
                f"./log_plots/{str(datetime.now())[:-7].replace(' ', '_')}_linfit_"
                + f"{stagename}_{movement_type.value}_{axis}-vs-{focus_axis}.png",
                format="png",
                facecolor="white",
                dpi=200,
            )
        if show_plot:
            plt.show(block=True)
        plt.close()

    return lin_result


def plane_grid(
    stage: StageDevices,
    movement_type: MovementType,
    plane: Distance,
    axes: list,
    axis0: np.ndarray,
    axis1: np.ndarray,
    exposure_time: Union[int, float],
    show_plot: bool = False,
    log_plot: bool = True,
):
    """
    Records the sensor value at a grid of values in a plane normal to the
    focus axis.

    Parameters
    ----------
    stage : `movement_classes.StageDevices`
        The stage for which to perform the grid search.
    movement_type : `movement_classes.MovementType`
        The movement type with which to move the fiber in the plane.
    plane : `Distance.Distance`
        The position of the plane along the focus axis.
    axes : str, list of str
        The pair of axes which the plane lies in. Used to infer the focus
        axis. Must have length 2. Also used for plotting.
    axis0, axis1 : np.ndarray
        The grid positions to visit for the 0th and 1st axes in `axes`,
        respectively.
    exposure_time : int, float
        The exposure time to use for reading the stage's sensor at each
        grid position.
    show_plot : bool, default=False
        Whether to display a plot of the grid search. Opens a new window
        containing the plot which must be closed for the program to
        continue.
    log_plot : bool, default=True
        Whether to save the plot of the grid search. The plot's filepath
        will be
        ``./log_plots/yyyy-mm-dd_hh:mm:ss_gridvalues_<stagename>_<movement_type>_<focus_axis>-{plane in microns}um.png``.
    """
    focus_axis = list(VALID_AXES.difference(set(axes)))[0]

    axis0_grid, axis1_grid = np.meshgrid(axis0, axis1)
    response_grid = np.zeros_like(axis0_grid)
    for i, pos0 in enumerate(axis0):
        stage.goto(axes[0], Distance(pos0, "microns"), movement_type)
        if i % 2:  # to snake along the grid
            for j, pos1 in enumerate(axis1):
                stage.goto(axes[1], Distance(pos1, "microns"), movement_type)
                response_grid[j, i] += stage.integrate(exposure_time)
        else:
            for j, pos1 in enumerate(axis1[::-1]):
                stage.goto(axes[1], Distance(pos1, "microns"), movement_type)
                response_grid[-j - 1, i] += stage.integrate(exposure_time)

    logger.info(f"Grid values:\n{response_grid}")

    if log_plot or show_plot:
        fig = plot_plane(response_grid, axis0_grid, axis1_grid, axes, plane)
        logger.info("Plot Generated")
        if log_plot:
            fig.savefig(
                f"./log_plots/{str(datetime.now())[:-7].replace(' ', '_')}_gridvalues_"
                + f"{stage.name}_{movement_type.value}_{focus_axis}-{plane.microns}um.png",
                format="png",
                facecolor="white",
                dpi=200,
            )
            logger.info("plot saved to ./log_plots")
        if show_plot:
            plt.show(block=True)
        plt.close()

    return response_grid


def run(
    stage: StageDevices,
    movement_type: MovementType,
    exposure_time: Union[int, float],
    spacing: Union[None, Distance, Sequence[Distance]] = None,  # default 10 volts
    num_points: Union[None, int, Sequence[int]] = None,
    center: Union[None, str, Sequence[Distance]] = None,
    limits: Optional[
        Sequence[Distance]
    ] = None,  # 2-list of (2-lists of) Distance objects
    axes: Union[List[str], str] = "xz",
    planes: Union[None, int, Sequence[Distance]] = None,  # default 3 planes
    fit_3d: bool = True,
    fit_2d: bool = True,
    fit_widths: bool = True,
    fit_lin: bool = True,
    max_pos: bool = True,
    no_fits: Optional[bool] = None,
    show_plots: Optional[bool] = True,
    log_plots: Optional[bool] = None,
    plane_kwargs: dict = {},
    fit_3d_kwargs: dict = {},
    fit_2d_kwargs: dict = {},
    fit_widths_kwargs: dict = {},
    fit_lin_kwargs: dict = {},
):
    """
    Executes a grid search in one or more planes normal to the focus axis.

    Parameters
    ----------
    stage : `movement_classes.StageDevices`
        The stage for which to perform the grid search.
    movement_type : `movement_classes.MovementType`
        The movement type with which to move the fiber in each plane. Note
        that movement along the focus axis is always performed with the
        stepper.
    exposure_time : int, float
        The exposure time to use for recording the stage's sensor value at
        each grid position.
    spacing : `Distance.Distance` or a sequence thereof, optional
        The distance between points in the grid. If a sequence, must
        contain one or two `Distance.Distance` objects. Providing two
        distance objects allows for defining a rectangular grid, in which
        case the two spacings are applied in order to the `axes`. If a
        single distance is supplied (on its own or in a single-element
        sequence), this spacing is applied to both axes. `spacing` and
        `num_points` cannot both be supplied; one must be ``None``. If both
        `spacing` and `num_points` are ``None`` then `spacing` defaults to
        15 volts if `movement_type` is ``PIEZO`` or 200 full steps if
        `movement_type` is ``STEPPER``.
    num_points : int or a sequence thereof, optional
        The number of points that should make up the grid within the
        `limits`. If a sequence, it must contain one or two ``int``s.
        Providing two ``int``s allows for defining a rectangular grid,
        which case the two values are applied in order to the `axes`. If a
        single int is supplied (on its own or in a single-element
        sequence), this value is applied to both axes. `spacing` and
        `num_points` cannot both be supplied; one must be ``None``. If both
        `spacing` and `num_points` are ``None`` then `spacing` defaults to
        15 volts if `movement_type` is ``PIEZO`` or 200 full steps if
        `movement_type` is ``STEPPER``.
    center : str, sequence of `Distance.Distance` objects, optional
        The position of the center of the grid in the planes. If `center`
        is "center", then the grid is centered on the center of the chosen
        movement types range of motion. If `center` is "current", then the
        grid is centered on the current stage position. A sequence of
        `Distance.Distance` objects of length 2 can also be used to choose
        an arbitrary center. The distance values are understood to be in
        the order of `axes`. ``None`` does not result in a default value,
        but instead results in any `limits` being relative to the chosen
        movement type's axis origin.
    limits : sequence of `Distance.Distance` objects, or a sequence thereof, optional
        Must be of shape (2) or (2,2). A single sequence is used to define
        the lower and upper limits of the grid for both axes
        simultaneously. A nested sequence allows for defining different
        limits for each of the axes, applied in the same order as `axes`.
        If `center` is ``None``, then the values of `limits` are taken to
        be "absolute", in that they take the origin to be the lower limit
        of the chosen movement type's range of motion. Otherwise, the
        values of `limits` are assumed to be relative to the center of the
        grid. ``None`` defaults to the limits of the chosen movement type
        (as in, the grid spans the entirety of the movement_type's range of
        motion).
    axes: str, list of str
        The pair of axes which the planes lie in. Used to infer the focus
        axis. Must have length 2. The order of the axes can affect the
        interpretation of other kwargs like `spacing`, `num_points`, and
        `limits`.
    planes : int, sequence of `Distance.Distance` objects, optional
        If an int is supplied, it is taken to be the number of planes to
        search in, evenly distributed along the focus axis's range of
        motion (stepper). Alternatively, a sequence of distance objects
        canbe supplied to explicitly determine the position of the planes
        in which to conduct a grid search. ``None`` defaults to 3.
    fit_3d, fit_2d, fit_widths, fit_lin : bool, default=True
        Whether to call `gaussbeam_fit_3d`, `gaussbeam_fit_2d`,
        `width_parafit`, or `peaks_linfit`, respectively, if possible. All
        of these are overridden if `no_fits` is not ``None``.
    max_pos : bool, default=True
        Whether to move the fiber to position at which the sensor recorded
        the highest intensity if all fits fail or are rejected.
    no_fits : bool, optional
        Convenience argument for overridding `fit_3d`, `fit_2d`,
        `fit_widths`, and `fit_lin` simultaneously. Has no effect if
        ``None``.
    show_plots : None, bool, default=True
        If not ``None``, updates (but does not override) the value of
        ``show_plot`` in `plane_kwargs`, `fit_3d_kwargs`, `fit_2d_kwargs`,
        `fit_widths_kwargs`, and `fit_lin_kwargs`.
    log_plots : None, bool, default=None
        If not ``None``, updates (but does not override) the value of
        ``log_plot`` in `plane_kwargs`, `fit_3d_kwargs`, `fit_2d_kwargs`,
        `fit_widths_kwargs`, and `fit_lin_kwargs`.
    plane_kwargs, fit_3d_kwargs, fit_2d_kwargs, fit_widths_kwargs,
        fit_lin_kwargs : dict, default={}
        For providing individual keyword arguments to `plane_grid`,
        `gaussbeam_fit_3d`, `gaussbeam_fit_2d`, `width_parafit`, and
        `peaks_linfit`, respectively. Not overridden by `show_plots` or
        `log_plots`. Currently, `show_plot` and `log_plot` are the only
        kwargs available for these functions.

    Examples
    --------
    The three non-keyword arguments are usually supplied by main.py, so they are not included in the examples below. Note that these examples are written to be valid python syntax, which does not exactly match the syntax used in the main menu of `main.py`.

    run(
    ..., MovementType.STEPPER, ...,
    spacing=Distance(10),
    center="center",
    limits=[Distance(-100),
    Distance(100)],
    axes="xz",
    planes=5
    )
    This will result in 21x21 grids of points, separated by 10um, centered
    in the steppers range of motion, in 5 xz-planes evenly distributed
    along the y-axis stepper's range of motion.

    Assuming that the steppers' range of motion is exactly 4000um, this
    could have also been achieved with the following call.
    run(
    ..., MovementType.STEPPER, ...,
    num_points=21,
    limits=[Distance(1900), Distance(2100)],
    axes="xz",
    planes=[Distance(0), Distance(1000), Distance(2000),
            Distance(3000), Distance(4000)]
    )
    Notice that the values of `limits` are different because we did not
    supply a value for `center`.

    run(
    ...,
    limits=[Distance(1500), Distance(1900)],
    num_points=[50, 20],
    axes="xz"
    )
    The above will result in a grid with a square search area but a
    rectangular grid pattern, with 50 points along the x axis but only 20
    points along the z axis.

    run(
    ...,
    limits=[[Distance(1300), Distance(1700)], [Distance(1000), Distance(3000)]],
    spacing=Distance(20),
    axes="yz"
    )
    In contrast, the above will result in a square-spaced grid over a
    rectangular search area.

    run(
    ...,
    spacing=Distance(50),
    num_points=20
    )
    This will fail an assertion, because `spacing` and `num_points` cannot
    be given together.

    Notes
    -----
    Below is the decision tree the function goes through once it has gathered all of the data from the sensor.

    If there is more than one plane, fit with 3d Gaussian beam, if enabled.
      - If accepted, go to 3d waist position and return.
    If 3d fit is not accepted or doesn't run, fit each plane with a 2d
    Gaussian, if enabled.
      - If at least 3 2d Gaussian fits were accepted, fit a parabola to the
    best-fit sigma values.
         -- If the parabola fit is accepted, fit each axis' best-fit peak
    position vs focus axis position with a line, if enabled.
            --- For each linear fit, if it is not accepted or didn't run,
    find the mean of the axis' peak  positions.
            --- Go to the focus of the fit-width position, then go to the
    position in that plane detemined by linear fit or taking the mean.
         -- If the parabola fit wasn't accepted or wasn't enabled, and
    there is at least one successful 2d Gaussian fit, go to the position of
    the highest best-fit peak.
      - If no 2d Gaussian fits were accepted or they didn't run, and
    fit_widths is enabled, determine the standard deviation of each plane
    and fit a parabola to those widths.
         -- If the parabola fit is accepted, fit each axis' photocenter
    position vs focus axis position with a line, if enabled.
            --- For each linear fit, if it is not accepted or didn't run,
    find the mean of the axis' peak photocenter positions.
            --- Go to the focus of the std-width position, then go to the
    position in that plane detemined by linear fit or or taking the mean.
    If no fitting is enabled (or just linear), then go to the position of the highest recorded intensity.
    """
    assert movement_type in (
        MovementType.PIEZO,
        MovementType.STEPPER,
    ), "movement_type must be MovementType.PIEZO or .STEPPER"
    #   may add .GENERAL later
    assert not (
        spacing is not None and num_points is not None
    ), "Cannot supply both spacing and num_points"

    axes = list(axes)
    assert (
        len(axes) == 2 and axes[0] in VALID_AXES and axes[1] in VALID_AXES
    ), "axes must be two of 'x', 'y', or 'z'"
    assert axes[0] != axes[1], "axes must be unique"
    focus_axis = list(VALID_AXES.difference(set(axes)))[0]

    # show_plots and log_plots will not override explicitly provided kwargs
    if show_plots is not None:
        for kwargs in (
            plane_kwargs,
            fit_3d_kwargs,
            fit_2d_kwargs,
            fit_widths_kwargs,
            fit_lin_kwargs,
        ):
            if "show_plot" not in kwargs.keys():
                kwargs.update({"show_plot": show_plots})
    if log_plots is not None:
        for kwargs in (
            plane_kwargs,
            fit_3d_kwargs,
            fit_2d_kwargs,
            fit_widths_kwargs,
            fit_lin_kwargs,
        ):
            if "log_plot" not in kwargs.keys():
                kwargs.update({"log_plot": log_plots})

    # no_fits will override any other given values of the fit bools
    if no_fits is not None:
        fit_3d = fit_2d = fit_widths = fit_lin = max_pos = not no_fits

    #   Default values
    if spacing is None and num_points is None:
        if movement_type == MovementType.PIEZO:
            spacing = Distance(15, "volts")
        if movement_type == MovementType.STEPPER:
            spacing = Distance(200, "fullsteps")
    if planes is None:
        planes = 3

    if limits is None:
        if movement_type == MovementType.PIEZO:
            limits_default = []
            for axis in axes:
                axis_center = stage.axes[axis].piezo_center
                lower, upper = stage.axes[axis].piezo_limits
                if center == "center":
                    lower -= axis_center
                    upper -= axis_center
                limits_default.append([lower, upper])
        elif movement_type == MovementType.STEPPER:
            limits_default = []
            for axis in axes:
                axis_center = stage.axes[axis].stepper_center
                lower, upper = stage.axes[axis].stepper_limits
                if center == "center":
                    lower -= axis_center
                    upper -= axis_center
                limits_default.append([lower, upper])
        else:
            raise ValueError(
                "movement_type must be a MovementType.PIEZO or " + ".STEPPER enum"
            )
        limits = limits_default

    if center == "center":
        if movement_type == MovementType.STEPPER:
            center = [stage.axes[axis].stepper_center for axis in axes]
        if movement_type == MovementType.PIEZO:
            center = [stage.axes[axis].piezo_center for axis in axes]
    if center == "current":
        if movement_type == MovementType.STEPPER:
            center = [stage.axes[axis].get_stepper_position() for axis in axes]
        if movement_type == MovementType.PIEZO:
            center = [stage.axes[axis].get_piezo_position() for axis in axes]
    if not isinstance(center, sequence) and center is not None:
        raise ValueError(
            "center must be None, 'center', 'current', or" + " a 2-tuple of Distances"
        )
    if center is not None and not all(isinstance(elem, Distance) for elem in center):
        raise ValueError(
            "center must be None, 'center', 'current', or" + " a 2-tuple of Distances"
        )
    center = cast(List[Distance], center)

    #   Duck typing
    if isinstance(limits, sequence) and all(
        isinstance(limit, Distance) for limit in limits
    ):
        limits = [copy.deepcopy(limits) for _ in range(2)]
    if isinstance(spacing, Distance):
        spacing = (spacing, spacing)
    if isinstance(num_points, int):
        num_points = (num_points, num_points)

    if isinstance(planes, int):
        planes_limits = [
            limit.microns for limit in stage.axes[focus_axis].stepper_limits
        ]
        planes_microns = np.linspace(planes_limits[0], planes_limits[1], planes)
        planes = [Distance(plane, "microns") for plane in planes_microns]

    if center is not None:
        limits[0][0] += center[0]
        limits[0][1] += center[0]
        limits[1][0] += center[1]
        limits[1][1] += center[1]

    axis0: np.ndarray = np.array([])
    axis1: np.ndarray = np.array([])

    if spacing is not None:
        axis0 = np.arange(
            limits[0][0].microns, (1 + 1e-6) * limits[0][1].microns, spacing[0].microns
        )
        axis1 = np.arange(
            limits[1][0].microns, (1 + 1e-6) * limits[1][1].microns, spacing[1].microns
        )
        # 1+1e-6 is so that the upper limit is included in the resulting list
    if num_points is not None:
        axis0 = np.linspace(limits[0][0].microns, limits[0][1].microns, num_points[0])
        axis1 = np.linspace(limits[1][0].microns, limits[1][1].microns, num_points[1])

    axis0_grid, axis1_grid = np.meshgrid(axis0, axis1)

    # take data in planes
    grid_values = []
    for plane in planes:
        logger.info(
            f"Running grid search in plane {focus_axis} = "
            + f"{plane.prettyprint()} microns with grid:\n"
            + f"{axes[0]} = {axis0} microns\n{axes[1]} = {axis1} microns\n"
            + "Total integration time ~ "
            + f"{sigfig.round(len(axis0) * len(axis1) * exposure_time / 1e3, 3)} seconds"
        )
        stage.goto(focus_axis, plane, MovementType.STEPPER)

        plane_values = plane_grid(
            stage,
            movement_type,
            plane,
            axes,
            axis0,
            axis1,
            exposure_time,
            **plane_kwargs,
        )
        grid_values.append(plane_values)

    # make cubes
    grid_values = np.array(grid_values)
    axis0_cube = np.tile(axis0_grid, (len(planes), 1, 1))
    axis1_cube = np.tile(axis1_grid, (len(planes), 1, 1))
    focus_cube = np.array(
        [plane.microns * np.ones_like(axis0_grid) for plane in planes]
    )

    if fit_3d:
        if len(planes) <= 1:
            logger.info(
                "There are not enough planes to do a 3d Gaussian beam model fit"
            )
        else:
            logger.info("Attempting to fit data with a 3d Gaussian beam model")

            result = gaussbeam_fit_3d(
                axes,
                movement_type,
                stage.name,
                axis0_cube,
                axis1_cube,
                focus_cube,
                grid_values,
                **fit_3d_kwargs,
            )

            accepted = True
            if not result.success:
                logger.info(
                    "Fit Rejected: 3d Gaussian beam fit did not finish successfully"
                )
                accepted = False
            elif fit_3d_kwargs["show_plot"]:
                accepted = accept_fit()

            if accepted:
                peak = result.params["I0"].value
                best_pos = [
                    Distance(result.params[f"waistx{n}"].value, "microns")
                    for n in range(1, 4)
                ]
                stage.goto(axes[0], best_pos[0], movement_type)
                stage.goto(axes[1], best_pos[1], movement_type)
                stage.goto(focus_axis, best_pos[2], movement_type)

                logger.info(
                    f"Moved to maximum of {peak} at ({axes[0]}, {axes[1]}, {focus_axis}) = "
                    + f"({best_pos[0].prettyprint()}, {best_pos[1].prettyprint()},"
                    + f"{best_pos[2].prettyprint()})"
                )
                return

    else:
        logger.debug("Skipped 3d fit")

    if fit_2d:
        logger.info("Attempting to fit planes with 2d Gaussian model")
        accepted_results = []
        accepted_planes = []
        for plane, plane_values in zip(planes, grid_values):
            result = gaussbeam_fit_2d(
                axes,
                movement_type,
                stage.name,
                axis0_grid,
                axis1_grid,
                plane_values,
                plane,
                **fit_2d_kwargs,
            )
            accepted = True
            if not result.success:
                logger.info(f"Fit to plane {focus_axis} = {plane.prettyprint()} failed")
                accepted = False
                continue

            if fit_2d_kwargs["show_plot"]:
                accepted = accept_fit()

            if accepted:
                accepted_results.append(result)
                accepted_planes.append(plane)

        # fit parabola to plane-fit widths if able
        if len(accepted_results) >= 3:
            logger.info("Fitting parabola to widths")
            widths = np.array(
                [result.params["sigmax"].value for result in accepted_results]
            )
            widths_unc = np.array(
                [result.params["sigmax"].stderr for result in accepted_results]
            )

            para_result = width_parafit(
                axes,
                movement_type,
                stage.name,
                accepted_planes,
                widths,
                widths_unc,
                **fit_widths_kwargs,
            )
            accepted = True
            if not para_result.success:
                logger.info(f"Quadratic fit to widths vs {focus_axis} failed")
                accepted = False

            elif fit_widths_kwargs["show_plot"]:
                accepted = accept_fit()

            focus_pos = -para_result.params["b"].value / (
                2 * para_result.params["a"].value
            )
            waist_pos = [None, None, Distance(focus_pos, "microns")]
            accepteds = [False, False]

            if accepted and fit_lin:
                logger.info("Fitting linear functions to peak center values")

                axis_peak_pos0 = np.array(
                    [result.params["centerx"].value for result in accepted_results]
                )
                axis_peak_unc0 = np.array(
                    [result.params["centerx"].stderr for result in accepted_results]
                )
                axis_peak_pos1 = np.array(
                    [result.params["centery"].value for result in accepted_results]
                )
                axis_peak_unc1 = np.array(
                    [result.params["centery"].stderr for result in accepted_results]
                )
                width_min = Distance(
                    cast(float, para_result.eval(x=focus_pos)), "microns"
                )
                lin_result0 = peaks_linfit(
                    axes,
                    True,
                    movement_type,
                    stage.name,
                    accepted_planes,
                    axis_peak_pos0,
                    axis_peak_unc0,
                    focus_pos,
                    **fit_lin_kwargs,
                )
                lin_result1 = peaks_linfit(
                    axes,
                    False,
                    movement_type,
                    stage.name,
                    accepted_planes,
                    axis_peak_pos1,
                    axis_peak_unc1,
                    focus_pos,
                    **fit_lin_kwargs,
                )

                lin_results = (lin_result0, lin_result1)
                for i, result in enumerate(lin_results):
                    accepteds[i] = True
                    if not result.success:
                        accepteds[i] = False
                    elif fit_lin_kwargs["show_plot"]:
                        accepteds[i] = accept_fit()
                    if accepteds[i]:
                        waist_pos[i] = Distance(
                            cast(SupportsFloat, result.eval(x=focus_pos)), "microns"
                        )

                for i, accept in enumerate(accepteds):
                    centerstr = "centerx" if i == 0 else "centery"
                    if not accept:
                        widthpos = np.mean(
                            [
                                result.params[centerstr].value
                                for result in accepted_results
                            ]
                        )
                        waist_pos[i] = Distance(widthpos, "microns")

                stage.goto(axes[0], waist_pos[0], movement_type)
                stage.goto(axes[1], waist_pos[1], movement_type)
                stage.goto(focus_axis, waist_pos[2], movement_type)

                logger.info(
                    f"Moved to minimum fit-width of {width_min.prettyprint()} at ({axes[0]}, {axes[1]}, {focus_axis}) = "
                    + f"({waist_pos[0].prettyprint()}, {waist_pos[1].prettyprint()}, "
                    + f"{waist_pos[2].prettyprint()})"
                )
                return

        # if any fits were successful, go to the highest known best-fit peak if wanted
        while True:
            print("Go to highest best-fit peak?")
            user_input = input("(y/n): ").strip().lower()
            if user_input == "y":
                highest_peak = True
                break
            if user_input == "n":
                highest_peak = False
                break
            print("Input must be 'y' or 'n'")

        if highest_peak and len(accepted_results) != 0:
            plane_peaks = np.array(
                [result.params["height"].value for result in accepted_results]
            )
            max_peak = plane_peaks.max()
            max_peak_idx = plane_peaks.argmax()

            best_plane = accepted_results[max_peak_idx]
            best_pos = [
                Distance(best_plane.params["centerx"].value, "microns"),
                Distance(best_plane.params["centery"].value, "microns"),
                accepted_planes[max_peak_idx],
            ]

            stage.goto(axes[0], best_pos[0], movement_type)
            stage.goto(axes[1], best_pos[1], movement_type)
            stage.goto(focus_axis, best_pos[2], movement_type)

            logger.info(
                f"Moved to maximum of known peaks {max_peak} at ({axes[0]}, {axes[1]}, {focus_axis}) = "
                + f"({best_pos[0].prettyprint()}, {best_pos[1].prettyprint()},"
                + f"{best_pos[2].prettyprint()})"
            )
            return

        logger.info("No planes were successfully fit with a gaussian")

    else:
        logger.debug("Skipped 2d fits")

    # fit parabola to plane standard deviations
    if fit_widths and len(planes) >= 3:
        background = stats.mode(grid_values.flatten)[0]
        logger.info(
            f"Sensor background determined to be {sigfig.round(background, 3)}"
            + "by taking the mode of all grid values"
        )
        grid_values_bg = grid_values - background
        widths = np.std(grid_values_bg, axis=(1, 2))
        logger.info(f"Plane standard deviations: {widths}")
        widths_unc = np.ones_like(widths)

        para_result = width_parafit(
            axes,
            movement_type,
            stage.name,
            planes,
            widths,
            widths_unc,
            **fit_widths_kwargs,
        )
        accepted = True
        if not para_result.success:
            logger.info(f"Quadratic fit to std-widths vs {focus_axis} failed")
            accepted = False

        elif fit_widths_kwargs["show_plot"]:
            accepted = accept_fit()

        focus_pos = -para_result.params["b"].value / 2 * para_result.params["a"].value
        waist_pos = [None, None, Distance(focus_pos, "microns")]
        accepteds = [False, False]

        if accepted and fit_lin:
            logger.info("Fitting linear functions to peak center values")

            plane_xgrid, plane_ygrid = np.meshgrid(*grid_values[0].shape)
            axis_peak_pos0 = []
            axis_peak_pos1 = []
            for grid in grid_values:
                axis_peak_pos0.append(np.sum(plane_xgrid * grid) / np.sum(plane_xgrid))
                axis_peak_pos1.append(np.sum(plane_ygrid * grid) / np.sum(plane_ygrid))

            axis_peak_pos0 = np.array(axis_peak_pos0)
            axis_peak_pos1 = np.array(axis_peak_pos1)
            axis_peak_unc0 = np.ones_like(axis_peak_pos0)
            axis_peak_unc1 = np.ones_like(axis_peak_pos1)

            width_min = Distance(
                cast(SupportsFloat, para_result.eval(x=focus_pos)), "microns"
            )
            lin_result0 = peaks_linfit(
                axes,
                True,
                movement_type,
                stage.name,
                planes,
                axis_peak_pos0,
                axis_peak_unc0,
                focus_pos,
                **fit_lin_kwargs,
            )
            lin_result1 = peaks_linfit(
                axes,
                False,
                movement_type,
                stage.name,
                planes,
                axis_peak_pos1,
                axis_peak_unc1,
                focus_pos,
                **fit_lin_kwargs,
            )

            lin_results = (lin_result0, lin_result1)
            for i, result in enumerate(lin_results):
                accepteds[i] = True
                if not result.success:
                    accepteds[i] = False
                elif fit_lin_kwargs["show_plot"]:
                    accepteds[i] = accept_fit()
                if accepteds[i]:
                    waist_pos[i] = Distance(
                        cast(SupportsFloat, result.eval(x=focus_pos)), "microns"
                    )

            axis_peak_poses = (axis_peak_pos0, axis_peak_pos1)
            for i, accept in enumerate(accepteds):
                centerstr = "centerx" if i == 0 else "centery"
                if not accept:
                    waistpos = np.mean(axis_peak_poses[i])
                    waist_pos[i] = Distance(waistpos, "microns")

            stage.goto(axes[0], waist_pos[0], movement_type)
            stage.goto(axes[1], waist_pos[1], movement_type)
            stage.goto(focus_axis, waist_pos[2], movement_type)

            logger.info(
                f"Moved to minimum fit-width of {width_min.prettyprint()} at ({axes[0]}, {axes[1]}, {focus_axis}) = "
                + f"({waist_pos[0].prettyprint()}, {waist_pos[1].prettyprint()}, "
                + f"{waist_pos[2].prettyprint()})"
            )
            return

    else:
        logger.debug("Skipped widths fitting")

    # go to position of brightest position visited
    if max_pos:
        max_value = grid_values.max()
        max_value_idx = np.unravel_index(grid_values.argmax(), grid_values.shape)
        max_value_pos = [
            Distance(axis0_cube[max_value_idx], "microns"),
            Distance(axis1_cube[max_value_idx], "microns"),
            Distance(focus_cube[max_value_idx], "microns"),
        ]

        stage.goto(axes[0], max_value_pos[0], movement_type)
        stage.goto(axes[1], max_value_pos[1], movement_type)
        stage.goto(focus_axis, max_value_pos[2], movement_type)

        logger.info(
            f"Moved to maximum of visited positions {max_value} at ({axes[0]}, {axes[1]}, {focus_axis}) = "
            + f"({max_value_pos[0].prettyprint()}, {max_value_pos[1].prettyprint()},"
            + f"{max_value_pos[2].prettyprint()})"
        )

    else:
        logger.debug("Skipped going to maximum position")

    return
