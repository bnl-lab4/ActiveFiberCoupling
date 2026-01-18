from typing import List, Union, cast

import lmfit
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from logging_utils import get_logger
from movement_classes import Distance

# unique logger name for this module
logger = get_logger(__name__)


VALID_AXES = {"x", "y", "z"}


def _plane_fit_string(fit_result, axes):
    """
    Generate a report on the 2D Gaussian fit in a plane. See `plot_2dfit`.
    """
    best_fit = []
    best_fit.append(f"A = {fit_result.uvars['height']}")
    best_fit.append(
        r"$\sigma_r$" + f" = {fit_result.uvars['sigmax']}" + r"$\mathrm{\mu m}$"
    )
    best_fit.append(
        f"{axes[0].upper()} = {fit_result.uvars['centerx']}" + r"$\mathrm{\mu m}$"
    )
    best_fit.append(
        f"{axes[1].upper()} = {fit_result.uvars['centery']}" + r"$\mathrm{\mu m}$"
    )
    best_fit.append(r"C" + f" = {fit_result.uvars['c']}")
    return "\n".join(best_fit)


def _para_fit_string(fit_result, axis):
    """
    Generate a report on the parabola fit to beam width vs distance. See `plot_para_fit`.
    """
    best_fit = []
    best_fit.append(f"$w({axis}) = a x^2 + b x + c$")
    best_fit.append("$a$" + f" = {fit_result.uvars['a']}" + r"$\mathrm{\mu m^{-1}}$")
    best_fit.append("$b$" + f" = {fit_result.uvars['b']}")
    best_fit.append("$c$" + f" = {fit_result.uvars['c']}" + r"$\mathrm{\mu m}$")
    return "\n".join(best_fit)


def _lin_fit_string(fit_result, axes):
    """
    Generate a report on the linear fit to peak position vs distance. See `plot_lin_fit`.
    """
    best_fit = []
    best_fit.append(f"$ {axes[0]} = m {axes[1]} + b $")
    best_fit.append("$m$" + f" = {fit_result.uvars['slope']}" + r"$\mathrm{\mu m}$")
    best_fit.append("$b$" + f" = {fit_result.uvars['intercept']}" + r"$\mathrm{\mu m}$")
    return "\n".join(best_fit)


def _gaussbeam_fit_string(fit_result, axes):
    """
    Generate a report of the 3D Gaussian beam fit. See `plot_3dfit`.
    """
    focus_axis = list(VALID_AXES.difference(set(axes)))[0]
    best_fit = []
    best_fit.append(f"$I_0$ = {fit_result.uvars['I0']}")
    best_fit.append(f"$w_0$ = {fit_result.uvars['w0']}" + r"$\mathrm{\mu m}$")
    best_fit.append(f"$C$ = {fit_result.uvars['C']}")
    best_fit.append(
        "$"
        + axes[0]
        + r"_\mathrm{waist}$"
        + f" = {fit_result.uvars['waistx1']}"
        + r"$\mathrm{\mu m}$"
    )
    best_fit.append(
        "$"
        + axes[1]
        + r"_\mathrm{waist}$"
        + f" = {fit_result.uvars['waistx2']}"
        + r"$\mathrm{\mu m}$"
    )
    best_fit.append(
        "$"
        + focus_axis
        + r"_\mathrm{waist}$"
        + f" = {fit_result.uvars['waistx3']}"
        + r"$\mathrm{\mu m}$"
    )
    best_fit = ", ".join(best_fit[:3]) + "\n" + ", ".join(best_fit[3:])
    return best_fit


def plot_2dfit(
    response_grid: np.ndarray,
    axis0_grid: np.ndarray,
    axis1_grid: np.ndarray,
    axes: Union[str, List[str]],
    plane: Distance,
    fit_result: lmfit.model.ModelResult,
) -> Figure:
    """
    Plot the result of fitting a circular Gaussian to a single plane.

    Plots the data, best fit, and residuals for the plane. The data and
    best fit plots share the same colormap, while the residual plot has an
    independent colormap. Displays the best fit values and uncertainties in
    the best fit plot.

    Parameters
    ----------
    response_grid, axis0_grid, axis1_grid : np.ndarray
        The sensor values and fiber positions in the grid, respectively.
        All three must have the same shape, with two dimensions.
    axes : str, list of strs
        The axis names corresponding to `axis0_grid` and `axis1_grid`, in
        that order. Must have length 2.
    plane : distance.Distance
        Distance object of the position of the plane in the third axis.
    fit_result : lmfit.model.ModelResult
        The result of the fil. Contains the sensor data, along with the best fit parameter values and uncertainties.

    Returns
    -------
    matplotlib.figure.Figure

    See Also
    --------
    grid_search.gaussbeam_fit_2d : Source of the fit results.
    """
    focus_axis = list(VALID_AXES.difference(set(axes)))[0]
    fig, axs = plt.subplots(
        figsize=(18, 5),
        nrows=1,
        ncols=5,
        layout="constrained",
        gridspec_kw=dict(width_ratios=(1, 0.05, 1, 1, 0.05)),
    )
    dense_axes = [
        np.linspace(axis.min(), axis.max(), 750) for axis in (axis0_grid, axis1_grid)
    ]
    dense_grids = np.meshgrid(*dense_axes)
    dense_result = fit_result.eval(x=dense_grids[0], y=dense_grids[1])
    result = fit_result.eval(x=axis0_grid, y=axis1_grid)
    residuals = response_grid - result
    vmin = min(response_grid.min(), dense_result.min())
    vmax = max(response_grid.max(), dense_result.max())

    data_c = axs[0].pcolormesh(
        axis0_grid, axis1_grid, response_grid, shading="auto", vmin=vmin, vmax=vmax
    )
    fig.colorbar(data_c, cax=axs[1], label="Intensity")
    axs[2].pcolormesh(*dense_grids, dense_result, shading="auto", vmin=vmin, vmax=vmax)
    residuals_c = axs[3].pcolormesh(axis0_grid, axis1_grid, residuals, shading="auto")
    fig.colorbar(residuals_c, cax=axs[4], label="Residual (Absolute)")

    titles = ("Data", "", "Best Fit", "Data - Best Fit")
    for ax, title in zip(axs, titles):
        if ax == axs[1]:
            continue
        ax.set_xlabel(axes[0] + " (microns)")
        ax.set_ylabel(axes[1] + " (microns)")
        ax.set_title(title)

    axs[2].sharey(axs[0])
    axs[2].tick_params(labelleft=False)
    axs[3].sharey(axs[0])
    axs[3].tick_params(labelleft=False)

    fig.suptitle(f"{focus_axis.upper()} = {plane.prettyprint()}")
    axs[2].annotate(
        _plane_fit_string(fit_result, axes),
        (0.0, 0.0),
        xytext=(0.01, 0.72),
        xycoords="axes fraction",
        annotation_clip=True,
        fontsize=10,
        color="white",
    )

    return fig


def plot_plane(
    response_grid: np.ndarray,
    axis0_grid: np.ndarray,
    axis1_grid: np.ndarray,
    axes: Union[str, List[str]],
    plane: Distance,
) -> Figure:
    """
    Plot the result of a grid search in a single plane.

    Parameters
    ----------
    response_grid, axis0_grid, axis1_grid : np.ndarray
        The sensor values and fiber positions in the grid, respectively.
        All three must have the same shape, with two dimensions.
    axes : str, list of strs
        The axis names corresponding to `axis0_grid` and `axis1_grid`, in
        that order. Must have length 2.
    plane : distance.Distance
        Distance object of the position of the plane in the third axis.

    Returns
    -------
    matplotlig.figure.Figure

    See Also
    --------
    grid_search.plane_grid : Takes the data in a single plane.
    """
    focus_axis = list(VALID_AXES.difference(set(axes)))[0]
    fig, data_ax = plt.subplots(layout="constrained")
    # pcolormesh correctly handles the cell centering for the given x and y arrays
    c = data_ax.pcolormesh(axis0_grid, axis1_grid, response_grid, shading="auto")
    fig.colorbar(c, ax=data_ax, label="Intensity")

    data_ax.set_xlabel(axes[0] + " (microns)")
    data_ax.set_ylabel(axes[1] + " (microns)")
    fig.suptitle(f"{focus_axis.upper()} = {plane.prettyprint()}", fontsize=10)

    return fig


def plot_para_fit(
    axes: Union[str, List[str]],
    widths: np.ndarray,
    widths_unc: np.ndarray,
    planes_microns: np.ndarray,
    result: lmfit.model.ModelResult,
) -> Figure:
    """
    Plot the result of fitting beam width vs focus-axis distance with a parabola.

    Displays the best fit values and uncertainties in the plot.

    Parameters
    ----------
    axes : str, list of strs
        The axis names corresponding to `axis0_grid` and `axis1_grid`, in
        that order. Must have length 2.
    widths, widths_unc : np.ndarray
        1D arrays containing the widths and width uncertainties of the best
        fit 2D Gaussians in each plane, respectively. Must have the same length.
    planes_microns : np.ndarray
        1D array containing the plane positions along the focal axis. Must
        have the same length as `widths` and `widths_unc`.
    result : lmfit.model.ModelResult
        The result of the fit. Contains the best fit parameter values and
        uncertainties.

    Returns
    -------
    matplotlib.figure.Figure

    See Also
    --------
    plot_2dfit : The source of the widths and their uncertainties.
    grid_search.width_parafit : Source of the fit results.
    """
    fake_unc = all(widths_unc == 1)
    focus_axis = list(VALID_AXES.difference(set(axes)))[0]
    planes_range = planes_microns.max() - planes_microns.min()
    ext_factor = 0.1
    dense_lims = (
        planes_microns.min() - ext_factor * planes_range,
        planes_microns.max() + ext_factor * planes_range,
    )
    planes_dense = np.linspace(*dense_lims, 1000)
    widths_dense = result.eval(x=planes_dense)
    fit_widths = result.eval(x=planes_microns)
    residuals = widths - fit_widths

    fig, axs = plt.subplots(
        nrows=2,
        figsize=(6, 6),
        layout="constrained",
        sharex=True,
        gridspec_kw=dict(height_ratios=(1, 0.3)),
    )
    axs[0].scatter(planes_microns, widths)
    if not fake_unc:
        axs[0].errorbar(planes_microns, widths, yerr=widths_unc, fmt="none")
    axs[0].plot(
        planes_dense,
        widths_dense,
        color="C1",
        label=_para_fit_string(result, focus_axis),
    )
    axs[1].scatter(planes_microns, residuals)
    if not fake_unc:
        axs[1].errorbar(planes_microns, residuals, yerr=widths_unc, fmt="none")

    axs[1].axhline(0, color="black", alpha=0.3)

    axs[0].legend()
    axs[0].grid(axis="both", which="both")
    axs[1].grid(axis="both", which="both")

    axs[1].set_xlabel(f"{focus_axis} (microns)")
    axs[0].set_ylabel(f"Width w({focus_axis}) (microns)")
    axs[1].set_ylabel("Residuals (Absolute)")

    return fig


def plot_3dfit(
    axes: Union[str, List[str]],
    axis0_cube: np.ndarray,
    axis1_cube: np.ndarray,
    focus_cube: np.ndarray,
    result: lmfit.model.ModelResult,
) -> Figure:
    """
    Plot the result of fitting a Gaussian beam to 3D data.

    For each focal plane, plots the data, best fit, and residuals in that
    plane. Note that the fit is performed over all planes simultaneously.
    The data and best fit plots have the same colormap for a given plane,
    while the residual plots all have independent colormaps. Displays the best
    fit values and uncertainties in the figure subtitle.

    axes : str, list of strs
        The axis names corresponding to `axis0_grid` and `axis1_grid`, in that order. Must have length 2.
    axis0_cube, axis1_cube, focus_cube : np.ndarray
        The the positions of the fiber along each axis. The `focus_cube`
        corresponds to the fiber position along the beam propagation
        direction, while `axis0_cube` and `axis1_cube` correspond to the
        two transverse axes. All three must have the same shape, with three
        dimensions.
    result : lmfit.model.ModelResult
        The result of the 3D fit. Contains the sensor data, along with the best fit parameter values and uncertainties.

    Returns
    -------
    matplotlib.figure.Figure

    See Also
    --------
    grid_search.gaussbeam_fit_3d : Source of the fit results.
    """
    focus_axis = list(VALID_AXES.difference(set(axes)))[0]
    data_cube = cast(np.ndarray, result.data).reshape(axis0_cube.shape)

    axis0_dense = np.arange(axis0_cube.min(), axis0_cube.max() + 1e-3, 1)
    axis1_dense = np.arange(axis1_cube.min(), axis1_cube.max() + 1e-3, 1)
    axis0_grid_dense, axis1_grid_dense = np.meshgrid(axis0_dense, axis1_dense)

    fig, axs = plt.subplots(
        nrows=len(focus_cube),
        ncols=5,
        figsize=((10, 3 * len(focus_cube) + 1)),
        layout="constrained",
        gridspec_kw=dict(width_ratios=(1, 0.05, 1, 1, 0.05)),
    )
    for axrow, grid_values, focus_grid in zip(axs, data_cube, focus_cube):
        plane = focus_grid[0, 0]
        focus_grid_dense = plane * np.ones_like(axis0_grid_dense)
        fit_grid_dense = result.eval(
            x1=axis0_grid_dense, x2=axis1_grid_dense, x3=focus_grid_dense
        )
        fit_grid = result.eval(x1=axis0_cube[0], x2=axis1_cube[0], x3=focus_grid)
        residuals_grid = grid_values - fit_grid

        vmin = min(grid_values.min(), fit_grid_dense.min())
        vmax = max(grid_values.max(), fit_grid_dense.max())

        data_cbar = axrow[0].pcolormesh(
            axis0_cube[0],
            axis1_cube[0],
            grid_values,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )
        axrow[2].pcolormesh(
            axis0_grid_dense,
            axis1_grid_dense,
            fit_grid_dense,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )
        residuals_cbar = axrow[3].pcolormesh(
            axis0_cube[0], axis1_cube[0], residuals_grid, shading="auto"
        )

        fig.colorbar(data_cbar, cax=axrow[1], label="Intensity")
        fig.colorbar(residuals_cbar, cax=axrow[4], label="Residual (Absolute)")

        axrow[2].sharey(axrow[0])
        axrow[2].tick_params(labelleft=False)
        axrow[3].sharey(axrow[0])
        axrow[3].tick_params(labelleft=False)

        axrow[0].set_title("Data")
        axrow[2].set_title(f"Fit\nPlane {focus_axis} = {plane:.0f} microns")
        axrow[3].set_title("Data - Fit")

        axrow[0].set_ylabel(axes[1] + " (microns)")
        for ax in (axrow[0], axrow[2], axrow[3]):
            ax.set_xlabel(axes[0] + " (microns)")

    for ax in axs[:-1, 0]:
        ax.sharex(axs[-1, 0])
    for ax in axs[:-1, 2]:
        ax.sharex(axs[-1, 2])
    for ax in axs[:-1, 3]:
        ax.sharex(axs[-1, 3])

    fig.suptitle(_gaussbeam_fit_string(result, axes))

    return fig


def plot_lin_fit(
    axis: str,
    focus_axis: str,
    axis_peak_pos: np.ndarray,
    axis_peak_unc: np.ndarray,
    planes_microns: np.ndarray,
    result: lmfit.model.ModelResult,
) -> Figure:
    """
    Plot the result of fitting beam position vs focus-axis distance with a line.

    Displays the best fit values and uncertainties in the plot. This plots
    the fit for only a single axis.

    Parameters
    ----------

    axes : str, list of strs
        The axis names corresponding to `axis0_grid` and `axis1_grid`, in
        that order. Must have length 2.
    axis_peak_pos, axis_peak_unc : np.ndarray
        1D arrays containing the peak positions and corresponding
        uncertainties of the best fit 2D Gaussians in each plane,
        respectively. Must have the same length.
    planes_microns : np.ndarray
        1D array containing the plane positions along the focal axis. Must
        have the same length as `axis_peak_pos` and `axis_peak_unc`.
    result : lmfit.model.ModelResult
        The result of the fit. Contains the best fit parameter values and
        uncertainties.

    Returns
    -------
    matplotlib.figure.Figure

    See Also
    --------
    grid_search.peaks_linfit : The source of the peak positions and uncertainties.
    """
    fake_unc = all(axis_peak_unc == 1)
    planes_range = planes_microns.max() - planes_microns.min()
    ext_factor = 0.1
    dense_lims = (
        planes_microns.min() - ext_factor * planes_range,
        planes_microns.max() + ext_factor * planes_range,
    )
    planes_dense = np.linspace(*dense_lims, 100)
    axis_peak_dense = result.eval(x=planes_dense)
    fit_axis_peak = result.eval(x=planes_microns)
    residuals = axis_peak_pos - fit_axis_peak

    fig, axs = plt.subplots(
        nrows=2,
        figsize=(6, 6),
        layout="constrained",
        sharex=True,
        gridspec_kw=dict(height_ratios=(1, 0.3)),
    )
    axs[0].scatter(planes_microns, axis_peak_pos)
    if not fake_unc:
        axs[0].errorbar(planes_microns, axis_peak_pos, yerr=axis_peak_unc, fmt="none")
    axs[0].plot(
        planes_dense,
        axis_peak_dense,
        color="C1",
        label=_lin_fit_string(result, axis + focus_axis),
    )
    axs[1].scatter(planes_microns, residuals)
    if not fake_unc:
        axs[1].errorbar(planes_microns, residuals, yerr=axis_peak_unc, fmt="none")

    axs[1].axhline(0, color="black", alpha=0.3)

    axs[0].legend()
    axs[0].grid(axis="both", which="both")
    axs[1].grid(axis="both", which="both")

    axs[1].set_xlabel(f"{focus_axis} (microns)")
    axs[0].set_ylabel(axis + " (microns)")
    axs[1].set_ylabel("Residuals (Absolute)")

    return fig
