########### TODO:
# deviation angles in 3d fit
###########

import numpy as np
import scipy.stats as stats
import lmfit
import copy
import sigfig
import matplotlib.pyplot as plt
from collections.abc import Sequence as sequence
from typing import Optional, Union, Sequence, MutableSequence, List, cast
from datetime import datetime

from MovementClasses import MovementType, StageDevices
from Distance import Distance
from grid_plotting import (
    plot_plane,
    plot_3dfit,
    plot_2dfit,
    plot_para_fit,
    plot_lin_fit,
)
from LoggingUtils import get_logger

# unique logger name for this module
logger = get_logger(__name__)

# max numpy arr size before it summarizes when printed
np.set_printoptions(threshold=2000, linewidth=250)

VALID_AXES = {"x", "y", "z"}
WAVELENGTH = 0.65  # microns


def accept_fit():
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
    pR2 = (
        np.pi * w0**2 / 0.65
    ) ** 2  # squared Rayleigh range in the propagation direction

    r2 = (x1 - waistx1) ** 2 + (
        x2 - waistx2
    ) ** 2  # squared distance from propagation axis
    p2 = (x3 - waistx3) ** 2  # squared position along propagation axis
    wp2 = w0**2 * (1 + p2 / pR2)  # squared waist at the given x3 position

    return I0 * (w0**2 / wp2) * np.exp(-2 * r2 / wp2) + C


def Gbeamfit_3d(
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


def Gbeamfit_2d(
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


def waist_parafit(
    axes: Union[str, List[str]],
    movement_type: MovementType,
    stagename: str,
    planes: Sequence[Distance],
    waists: np.ndarray,
    waists_unc: np.ndarray,
    show_plot: bool = False,
    log_plot: bool = True,
):
    focus_axis = list(VALID_AXES.difference(set(axes)))[0]
    planes_microns = np.array([plane.microns for plane in planes])

    model = lmfit.models.QuadraticModel()
    params = model.guess(waists, x=planes_microns)
    para_result = model.fit(
        waists, x=planes_microns, params=params, weights=1 / waists_unc
    )

    if show_plot or log_plot:
        fig = plot_para_fit(axes, waists, waists_unc, planes_microns, para_result)
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
    # fit peak position values (in case the beam is not parallel with the focal axis)
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
    exposureTime: Union[int, float],
    show_plot: bool = False,
    log_plot: bool = True,
):
    focus_axis = list(VALID_AXES.difference(set(axes)))[0]

    axis0_grid, axis1_grid = np.meshgrid(axis0, axis1)
    response_grid = np.zeros_like(axis0_grid)
    for i, pos0 in enumerate(axis0):
        stage.goto(axes[0], Distance(pos0, "microns"), movement_type)
        if i % 2:  # to snake along the grid
            for j, pos1 in enumerate(axis1):
                stage.goto(axes[1], Distance(pos1, "microns"), movement_type)
                response_grid[j, i] += stage.integrate(exposureTime)
        else:
            for j, pos1 in enumerate(axis1[::-1]):
                stage.goto(axes[1], Distance(pos1, "microns"), movement_type)
                response_grid[-j - 1, i] += stage.integrate(exposureTime)

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
    exposureTime: Union[int, float],
    spacing: Union[None, Distance, Sequence[Distance]] = None,  # default 10 volts
    num_points: Union[None, int, Sequence[int]] = None,
    center: Union[None, str, Sequence[Distance]] = None,
    limits: Optional[
        Sequence[Distance]
    ] = None,  # 2-list of (2-lists of) Distance objects
    axes: str = "xz",
    planes: Union[None, int, Sequence[Distance]] = None,  # default 3 planes
    fit_3d: bool = True,
    fit_2d: bool = True,
    fit_waists: bool = True,
    fit_lin: bool = True,
    max_pos: bool = True,
    no_fits: Optional[bool] = None,
    show_plots: Optional[bool] = True,
    log_plots: Optional[bool] = None,
    plane_kwargs: dict = {},
    fit_3d_kwargs: dict = {},
    fit_2d_kwargs: dict = {},
    fit_waists_kwargs: dict = {},
    fit_lin_kwargs: dict = {},
):
    assert movement_type in (MovementType.PIEZO, MovementType.STEPPER), (
        "movement_type must be MovementType.PIEZO or .STEPPER"
    )
    #   may add .GENERAL later
    assert not (spacing is not None and num_points is not None), (
        "Cannot supply both spacing and num_points"
    )

    axes = list(axes)
    assert len(axes) == 2 and axes[0] in VALID_AXES and axes[1] in VALID_AXES, (
        "axes must be two of 'x', 'y', or 'z'"
    )
    assert axes[0] != axes[1], "axes must be unique"
    focus_axis = list(VALID_AXES.difference(set(axes)))[0]

    # show_plots and log_plots will not override explicitly provided kwargs
    if show_plots is not None:
        for kwargs in (
            plane_kwargs,
            fit_3d_kwargs,
            fit_2d_kwargs,
            fit_waists_kwargs,
            fit_lin_kwargs,
        ):
            if "show_plot" not in kwargs.keys():
                kwargs.update({"show_plot": show_plots})
    if log_plots is not None:
        for kwargs in (
            plane_kwargs,
            fit_3d_kwargs,
            fit_2d_kwargs,
            fit_waists_kwargs,
            fit_lin_kwargs,
        ):
            if "log_plot" not in kwargs.keys():
                kwargs.update({"log_plot": log_plots})

    # no_fits will override any other given values of the fit bools
    if no_fits is not None:
        fit_3d = fit_2d = fit_waists = fit_lin = max_pos = not no_fits

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
                axis_center = stage.axes[axis].PIEZO_CENTER
                lower, upper = stage.axes[axis].PIEZO_LIMITS
                if center == "center":
                    lower -= axis_center
                    upper -= axis_center
                limits_default.append([lower, upper])
        elif movement_type == MovementType.STEPPER:
            limits_default = []
            for axis in axes:
                axis_center = stage.axes[axis].STEPPER_CENTER
                lower, upper = stage.axes[axis].STEPPER_LIMITS
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
            center = [stage.axes[axis].STEPPER_CENTER for axis in axes]
        if movement_type == MovementType.PIEZO:
            center = [stage.axes[axis].PIEZO_CENTER for axis in axes]
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
        limits = [copy.deepcopy(limits) for i in range(2)]
    if isinstance(spacing, Distance):
        spacing = (spacing, spacing)
    if isinstance(num_points, int):
        num_points = (num_points, num_points)

    if isinstance(planes, int):
        planes_limits = [
            limit.microns for limit in stage.axes[focus_axis].STEPPER_LIMITS
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
            + f"{sigfig.round(len(axis0) * len(axis1) * exposureTime / 1e3, 3)} seconds"
        )
        stage.goto(focus_axis, plane, MovementType.STEPPER)

        plane_values = plane_grid(
            stage,
            movement_type,
            plane,
            axes,
            axis0,
            axis1,
            exposureTime,
            **plane_kwargs,
        )
        grid_values.append(plane_values)

    """
    Logical flow:
    If there is more than one plane, fit with 3d Gaussian beam if enabled.
        If accepted, go to 3d waist position.
    If 3d fit is not accepted or doesn't run, fit each plane with a 2d Gaussian if enabled.
    If at least 3 2d Gaussian fits were accept, fit a parabola to the best-fit sigma values.
        If the parabola fit is accepted, fit each axis' best-fit peak position vs
                                            focus axis position with a line, if enabled.
            For each linear fit, if it is not accepted or didn't run, find the mean of the axis' peak  positions.
            Go to the focus of the fit-waist position, then go to the position in that plane detemined
                                            by linear fit or or taking the mean.
        If the parabola fit wasn't accepted or wasn't enabled, and there is at least one successful
                                            2d Gaussian fit, go to the position of the highest best-fit peak.
    If no 2d Gaussian fits were accepted or they didn't run, and fit_waists is enabled,
                                            determine the standard deviation of each plane and fit a
                                            parabola to those waists.
        If the parabola fit is accepted, fit each axis' photocenter position vs
                                            focus axis position with a line, if enabled.
            For each linear fit, if it is not accepted or didn't run, find the mean of the
                                            axis' peak photocenter positions.
            Go to the focus of the std-waist position, then go to the position in that plane detemined
                                            by linear fit or or taking the mean.
    If no fitting is enabled (or just linear), then go to the position of the highest recorded intensity.
    """

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

            result = Gbeamfit_3d(
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
            result = Gbeamfit_2d(
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

        # fit parabola to plane-fit waists if able
        if len(accepted_results) >= 3:
            logger.info("Fitting parabola to waists")
            waists = np.array(
                [result.params["sigmax"].value for result in accepted_results]
            )
            waists_unc = np.array(
                [result.params["sigmax"].stderr for result in accepted_results]
            )

            para_result = waist_parafit(
                axes,
                movement_type,
                stage.name,
                accepted_planes,
                waists,
                waists_unc,
                **fit_waists_kwargs,
            )
            accepted = True
            if not para_result.success:
                logger.info(f"Quadratic fit to waists vs {focus_axis} failed")
                accepted = False

            elif fit_waists_kwargs["show_plot"]:
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
                waist_min = Distance(
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
                        waist_pos[i] = Distance(result.eval(x=focus_pos), "microns")

                for i, accept in enumerate(accepteds):
                    centerstr = "centerx" if i == 0 else "centery"
                    if not accept:
                        waistpos = np.mean(
                            [
                                result.params[centerstr].value
                                for result in accepted_results
                            ]
                        )
                        waist_pos[i] = Distance(waistpos, "microns")

                stage.goto(axes[0], waist_pos[0], movement_type)
                stage.goto(axes[1], waist_pos[1], movement_type)
                stage.goto(focus_axis, waist_pos[2], movement_type)

                logger.info(
                    f"Moved to minimum fit-waist of {waist_min.prettyprint()} at ({axes[0]}, {axes[1]}, {focus_axis}) = "
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
    if fit_waists and len(planes) >= 3:
        background = stats.mode(grid_values.flatten)[0]
        logger.info(
            f"Sensor background determined to be {sigfig.round(background, 3)}"
            + "by taking the mode of all grid values"
        )
        grid_values_bg = grid_values - background
        waists = np.std(grid_values_bg, axis=(1, 2))
        logger.info(f"Plane standard deviations: {waists}")
        waists_unc = np.ones_like(waists)

        para_result = waist_parafit(
            axes,
            movement_type,
            stage.name,
            planes,
            waists,
            waists_unc,
            **fit_waists_kwargs,
        )
        accepted = True
        if not para_result.success:
            logger.info(f"Quadratic fit to std-waists vs {focus_axis} failed")
            accepted = False

        elif fit_waists_kwargs["show_plot"]:
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

            waist_min = Distance(para_result.eval(x=focus_pos), "microns")
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
                    waist_pos[i] = Distance(result.eval(x=focus_pos), "microns")

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
                f"Moved to minimum fit-waist of {waist_min.prettyprint()} at ({axes[0]}, {axes[1]}, {focus_axis}) = "
                + f"({waist_pos[0].prettyprint()}, {waist_pos[1].prettyprint()}, "
                + f"{waist_pos[2].prettyprint()})"
            )
            return

    else:
        logger.debug("Skipped waists fitting")

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
