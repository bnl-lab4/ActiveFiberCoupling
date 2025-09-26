########### TODO:
#   better search for focus
###########

import logging
import numpy as np
import lmfit
import copy
import matplotlib.pyplot as plt
from collections.abc import Sequence as sequence
from typing import Optional, Union, Tuple, Sequence
from datetime import datetime

from MovementClasses import MovementType, StageDevices, Distance

# unique logger name for this module
log = logging.getLogger(__name__)
VALID_AXES = {'x', 'y', 'z'}


def plot_plane(response_grid: np.array, axis0_grid: np.array, axis1_grid: np.array,
               axes: list, fit_result = None):
    if fit_result is None:
        fig, data_ax = plt.subplots()
        # pcolormesh correctly handles the cell centering for the given x and y arrays
        c = data_ax.pcolormesh(axis0_grid, axis1_grid, response_grid, shading='auto')
        fig.colorbar(c, ax=data_ax, label='Intensity')

        data_ax.set_xlabel(axes[0] + ' (microns)')
        data_ax.set_ylabel(axes[1] + ' (microns)')
        data_ax.set_title('Data')

        plt.show(block=True)
        return

    fig, axs = plt.subplots(figsize=(18, 5), nrows = 1, ncols = 5, layout = 'constrained',
                            gridspec_kw = dict(width_ratios = (1, 0.05, 1, 1, 0.05)))
    dense_axes = [np.linspace(axis.min(), axis.max(), 750) for axis in
                    (axis0_grid, axis1_grid)]
    dense_grids = np.meshgrid(*dense_axes)
    dense_result = fit_result.eval(x = dense_grids[0], y = dense_grids[1])
    result = fit_result.eval(x = axis0_grid, y = axis1_grid)
    resid = response_grid - result
    vmin = min(response_grid.min(), dense_result.min())
    vmax = max(response_grid.max(), dense_result.max())

    data_c = axs[0].pcolormesh(axis0_grid, axis1_grid, response_grid,
                           shading='auto', vmin=vmin, vmax=vmax)
    fig.colorbar(data_c, cax=axs[1], label='Intensity')
    axs[2].pcolormesh(*dense_grids, dense_result,
                           shading='auto', vmin=vmin, vmax=vmax)
    resid_c = axs[3].pcolormesh(axis0_grid, axis1_grid, resid, shading='auto')
    fig.colorbar(resid_c, cax=axs[4], label='Residual Intensity (Absolute)')

    titles = ('Data', '', 'Best Fit', 'Data - Best Fit')
    for ax, title in zip(axs, titles):
        if ax == axs[1]:
            continue
        ax.set_xlabel(axes[0] + ' (microns)')
        ax.set_ylabel(axes[1] + ' (microns)')
        ax.set_title(title)

    return fig


def grid_search(stage: StageDevices, movementType: MovementType,
                axis0: np.array, axis1: np.array, axes: list,
                exposureTime: Union[int, float], avg: bool = True,
                fit: bool = True, show_plot: bool = False, log_plot: bool = True):

    focus_axis = list(VALID_AXES.difference(set(axes)))[0]

    axis0_grid, axis1_grid = np.meshgrid(axis0, axis1)
    response_grid = np.zeros_like(axis0_grid)
    for i, pos0 in enumerate(axis0):
        stage.goto(axes[0], Distance(pos0, "microns"), movementType)
        for j, pos1 in enumerate(axis1):
            stage.goto(axes[1], Distance(pos1, "microns"), movementType)
            response_grid[j, i] += stage.integrate(exposureTime, avg)

    log.info(f"Grid values:\n{response_grid}")
    if not fit:
        maximum = response_grid.max()
        maximum_pos_idx = response_grid.argmax()
        maximum_pos_idx = np.unravel_index(maximum_pos_idx, response_grid.shape)
        maximum_pos = (axis0_grid[maximum_pos_idx], axis1_grid[maximum_pos_idx])
        width = None
        return maximum, maximum_pos, width

    gmodel = lmfit.models.Gaussian2dModel()
    cmodel = lmfit.models.ConstantModel()
    model = gmodel + cmodel
    params = gmodel.guess(response_grid.flatten(), axis0_grid.flatten(), axis1_grid.flatten())
    params['sigmay'].set(expr = 'sigmax')
    params.add('c', value = response_grid.min())

    #   normalize weights, assuming that more light is in the positive direction
    weights = response_grid
    weights = weights - weights.min() + 1e-3 * np.ones_like(weights)
    weights /= weights.max()
    weights = np.sqrt(weights)

    result = model.fit(response_grid, x = axis0_grid, y = axis1_grid,
                       params = params, weights = weights, nan_policy = 'omit')
    maximum = result.params['height'].value
    maximum_pos = (result.best_values['centerx'], result.best_values['centery'])
    maximum_pos_distance = [Distance(pos, "microns").prettyprint() for pos in maximum_pos]
    width = result.params['sigmax'].value
    width_distance = Distance(width, 'microns').prettyprint()
    log.info(f"Best fit peak of {maximum} at" +
            f"{maximum_pos_distance} with width {width_distance}")

    log.info("Plot Generated")
    fig = plot_plane(response_grid, axis0_grid, axis1_grid, axes, result)
    if log_plot:
        if movementType == MovementType.PIEZO:
            plane = stage.axes[focus_axis].get_piezo_position()
        if movementType == MovementType.STEPPER:
            plane = stage.axes[focus_axis].get_stepper_position()
        fig.savefig(f"./log_plots/{str(datetime.now())[:-7].replace(' ', '_')}_" +
            f"{movementType.value}_{focus_axis}-{plane.prettyprint()}.png")
    if show_plot():
        plt.show(block=True)

    return maximum, maximum_pos, width


def run(stage: StageDevices, movementType: MovementType, exposureTime: Union[int, float],
        spacing: Union[None, Distance, Sequence[Distance]] = None,  # default 10 volts
        num_points: Tuple[None, int, Sequence[int]] = None,
        center: Union[None, str, Sequence[Distance]] = None,
        limits: Optional[list] = None, # must be mutable; 2-list of (2-lists of) Distance objects
        axes: str = 'yz', planes: Union[None, int, Sequence[Distance]] = None,   # default 3 planes
        grid_search_kwargs: dict = {}):

    assert movementType in (MovementType.PIEZO, MovementType.STEPPER), \
            "movementType must be MovementType.PIEZO or .STEPPER"
            #   may add .GENERAL later
    assert not (spacing is not None and num_points is not None), \
            "Cannot supply both spacing and num_points"

    axes = list(axes)
    assert len(axes) == 2 and axes[0] in VALID_AXES and axes[1] in VALID_AXES, \
            "axes must be two of 'x', 'y', or 'z'"
    assert axes[0] != axes[1], "axes must be unique"
    focus_axis = list(VALID_AXES.difference(set(axes)))[0]

    #   Default values
    if spacing is None and num_points is None:
        if movementType == MovementType.PIEZO:
            spacing = Distance(15, "volts")
        if movementType == MovementType.STEPPER:
            spacing = Distance(200, "fullsteps")
    if planes is None:
        planes = 3

    if limits is None:
        if movementType == MovementType.PIEZO:
            limits = [copy.deepcopy(stage.axes[axis].PIEZO_LIMITS) for axis in axes]
        elif movementType == MovementType.STEPPER:
            limits = [copy.deepcopy(stage.axes[axis].STEPPER_LIMITS) for axis in axes]
        else:
            raise ValueError("movementType must be a MovementType.PIEZO or " +
                             ".STEPPER enum")

    if center == 'center':
        center = [stage.axes[axis].STEPPER_CENTER for axis in axes]
    if center == 'current':
        center = [stage.axes[axis].get_stepper_position() for axis in axes]

    if not isinstance(center, sequence) and center is not None:
        raise ValueError("center must be None, 'center', 'current', or" +
                         " a 2-tuple of Distances")
    if center is not None and \
            not all(isinstance(elem, Distance) for elem in center):
        raise ValueError("center must be None, 'center', 'current', or" +
                         " a 2-tuple of Distances")

    #   Duck typing
    if isinstance(limits, list) and all(isinstance(limit, Distance) for limit in limits):
        limits = [copy.deepcopy(limits) for i in range(2)]
    if isinstance(spacing, Distance):
        spacing = (spacing, spacing)
    if isinstance(num_points, int):
        num_points = (num_points, num_points)

    if isinstance(planes, int):
        if movementType == MovementType.PIEZO:
            planes = tuple(np.linspace(*stage.axes[focus_axis].PIEZO_LIMITS, planes))
        elif movementType == MovementType.STEPPER:
            planes = tuple(np.linspace(*stage.axes[focus_axis].STEPPER_LIMITS, planes))

    if center is not None:
        limits[0][0] += center[0]
        limits[0][1] += center[0]
        limits[1][0] += center[1]
        limits[1][1] += center[1]

    if spacing is not None:
        axis0 = np.arange(limits[0][0].microns, 1.1*limits[0][1].microns, spacing[0].microns)
        axis1 = np.arange(limits[1][0].microns, 1.1*limits[1][1].microns, spacing[1].microns)
    if num_points is not None:
        axis0 = np.linspace(limits[0][0].microns, limits[0][1].microns, num_points)
        axis1 = np.linspace(limits[1][0].microns, limits[1][1].microns, num_points)

    #   grid search in planes
    plane_maxima = []
    plane_maxima_pos = []
    widths = []
    for plane in planes:
        log.info(f"Running grid search in plane {focus_axis}={plane.prettyprint()} microns with grid:\n" +
                    f"{axes[0]} = {axis0} microns\n{axes[1]} = {axis1} microns")
        stage.goto(focus_axis, plane, movementType)
        maximum, maximum_pos, width = grid_search(stage, movementType,
                                    axis0, axis1, axes, exposureTime, **grid_search_kwargs)
        plane_maxima.append(maximum)
        plane_maxima_pos.append(maximum_pos)
        widths.append(width)

        maximum_pos = [Distance(pos, "microns").prettyprint() for pos in maximum_pos]
        log.info(f"{focus_axis} = {plane.microns} intensity maximum: {maximum}")
        log.info(f"{focus_axis} = {plane.microns} maximum at ({axes[0]}, {axes[1]}) = " +
                f"({maximum_pos[0]}, {maximum_pos[1]})")
        if width is not None:
            log.info(f"{focus_axis} = {plane.microns} gaussian width sigma = " +
                        Distance(width, 'microns').prettyprint())

    plane_maxima = np.array(plane_maxima)
    plane_maxima_pos = np.array(plane_maxima_pos)
    widths = np.array(widths)

    #   Determine optimal position in 3d space
    if len(planes) == 1:
        stage.goto(axes[0], Distance(plane_maxima_pos[0][0], "microns"), movementType)
        stage.goto(axes[1], Distance(plane_maxima_pos[0][1], "microns"), movementType)
        print("Done!")
        return

    if len(planes): # >= 2:
        maximum = plane_maxima.max()
        maximum_pos = plane_maxima_pos[plane_maxima.argmax()]
        maximum_pos = [Distance(pos, 'microns') for pos in maximum_pos]
        maximum_plane = planes[plane_maxima.argmax()]
        stage.goto(axes[0], maximum_pos[0], movementType)
        stage.goto(axes[1], maximum_pos[1], movementType)
        stage.goto(focus_axis, maximum_plane, movementType)
        log.info(f"Moved to maximum of {maximum} at ({focus_axis}, {axes[0]}, {axes[1]}) = " +
                f"({maximum_plane.prettyprint()}, {maximum_pos[0].prettyprint()}," +
                f"{maximum_pos[1].prettyprint()})")
        return

    #   real fitting for the 3d best point, either:
    #   measure points along the best fit line to the maxima
    #   find best fit to the beam waist, given planar widths
    #   true 3d fit to beam shape (requires grid_search to spit back more)
