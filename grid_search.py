########### TODO:
#   better search for focus
#   plotting function
###########

import logging
import numpy as np
import lmfit
import matplotlib.pyplot as plt
from typing import Optional, Union, Tuple

from MovementClasses import MovementType, StageDevices, StageAxis, Distance

# unique logger name for this module
log = logging.getLogger(__name__)
VALID_AXES = {'x', 'y', 'z'}


def plot_plane(response_grid: np.array, axis0_grid: np.array, axis1_grid: np.array,
               axes: list, fit_result = None, block: bool = True):  # idk how to type hint ModelResult
    if fit_result is None:
        fig, data_ax = plt.subplots()
        # pcolormesh correctly handles the cell centering for the given x and y arrays
        c = data_ax.pcolormesh(axis0_grid, axis1_grid, response_grid, shading='auto')
        fig.colorbar(c, ax=data_ax, label='Intensity')

        data_ax.set_xlabel(axes[0] + ' (microns)')
        data_ax.set_ylabel(axes[1] + ' (microns)')
        data_ax.set_title('Data')

        fig.show(block)
        return

    fig, axs = plt.subplots(figsize=(18, 5), nrows = 1, ncols = 5, layout = 'constrained',
                            gridspec_kw = dict(width_ratios = (1, 0.05, 1, 1, 0.05)))
    dense_axes = [np.linspace(axis.min(), axis.max(), 750) for axis in
                    (axis0_grid, axis1_grid)]
    dense_grids = np.meshgrid(*dense_axes)
    dense_result = fit_result.eval(x = dense_grids[0], y = dense_grids[1])
    result = fit_result.eval(x = axis0_grid, y = axis1_grid)
    resid = response_grid - result
    vmin = min(response_grid.min(), result.min())
    vmax = max(response_grid.max(), result.max())

    data_c = axs[0].pcolormesh(axis0_grid, axis1_grid, response_grid,
                           shading='auto', vmin=vmin, vmax=vmax)
    fig.colorbar(data_c, cax=axs[1], label='Intensity')
    axs[2].pcolormesh(*dense_grids, dense_result,
                           shading='auto', vmin=vmin, vmax=vmax)
    resid_c = axs[3].pcolormesh(axis0_grid, axis1_grid, resid, shading='auto')
    fig.colorbar(resid_c, cax=axs[4], label='Residual Intensity')

    titles = ('Data', '', 'Best Fit', 'Data - Best Fit')
    for ax, title in zip(axs, titles):
        if ax == axs[1]:
            continue
        ax.set_xlabel(axes[0] + ' (microns)')
        ax.set_ylabel(axes[1] + ' (microns)')
        ax.set_title(title)

    fig.show(block)
    return


def grid_search(stage: StageDevices, movementType: MovementType,
                axis0: np.array, axis1: np.array, axes: list,
                exposureTime: Union[int, float],
                avg: bool = True, fit: bool = True, plot: bool = True, block: bool = True):
    axis0_grid, axis1_grid = np.meshgrid(axis0, axis1)
    response_grid = np.zeros_like(axis0_grid)
    for i, pos0 in enumerate(axis0):
        stage.goto(axes[0], Distance(pos0, "microns"), movementType)
        for j, pos1 in enumerate(axis1):
            stage.goto(axes[1], Distance(pos1, "microns"), movementType)
            response_grid[j, i] += stage.integrate(exposureTime, avg)

    log.info(f"Grid values: {response_grid}")
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
    weights -= weights.min() + 1e-3 * np.ones_like(weights)
    weights / weights.max()
    weights = 1 / np.sqrt(response_grid)

    result = model.fit(response_grid, x = axis0_grid, y = axis1_grid,
                       params = params, weights = weights, nan_policy = 'omit')
    maximum = result.params['height'].value
    maximum_pos = (result.best_values['centerx'], result.best_values['centery'])
    width = result.params['sigmax'].value
    log.info(f"Best fit peak of {maximum} at {maximum_pos} with width {width} (microns)")

    if plot:
        log.info("Plot Generated")
        plot_plane(response_grid, axis0_grid, axis1_grid, axes, result, block)

    return maximum, maximum_pos, width


def run(stage: StageDevices, movementType: MovementType, exposureTime: Union[int, float],
        spacing: Union[None, Distance, Tuple[Distance, Distance]] = None,  # default 10 volts
        num_points: Tuple[None, int, Tuple[int, int]] = None,
        limits: Optional[Tuple] = None,                 # tuple of (tuples of) Distance objects
        axes: str = 'yz', planes: Union[None, int, Tuple[Distance, ...]] = None,
        grid_search_kwargs: Optional[dict] = None, fake_data: bool = False):  # default 3 planes

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
            spacing = Distance(10, "volts")
        if movementType == MovementType.STEPPER:
            spacing = Distance(200 * 32, "steps")
    if planes is None:
        planes = 3

    if limits is None:
        if movementType == MovementType.PIEZO:
            limits = (stage.PIEZO_LIMITS, ) * 2
        elif movementType == MovementType.STEPPER:
            limits = [stage.axes[grid_axis].STEPPER_LIMITS for grid_axis in axes]

    #   Duck typing
    if isinstance(spacing, Distance):
        spacing = (spacing, spacing)
    if isinstance(num_points, int):
        num_points = (num_points, num_points)

    if isinstance(planes, int):
        if movementType == MovementType.PIEZO:
            planes = tuple(np.linspace(*StageAxis.PIEZO_LIMITS, planes))
        elif movementType == MovementType.STEPPER:
            planes = tuple(np.linspace(*stage.axes[focus_axis].STEPPER_LIMITS, planes))

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
        log.info(f"Running grid search in plane {focus_axis}={plane.microns} microns with grid:\n"
                    f"{axes[0]} = {axis0} microns\n{axes[1]} = {axis1} microns")
        stage.goto(focus_axis, plane, movementType)
        maximum, maximum_pos, width = grid_search(stage, movementType,
                                 axis0, axis1, axes, exposureTime, **grid_search_kwargs)
        plane_maxima.append(maximum)
        plane_maxima_pos.append(maximum_pos)
        widths.append(width)

        log.info(f"{focus_axis} = {plane.microns} intensity maximum: {maximum}")
        log.info(f"{focus_axis} = {plane.microns} maximum at ({axes[0]}, {axes[1]}) = " +
                "({maximum_pos[0].microns}, {maximum_pos[1].microns})")
        if width is not None:
            log.info(f"{focus_axis} = {plane.microns} gaussian width sigma = {width}")

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
        maximum_plane = planes[plane_maxima.argmax()]
        stage.goto(axes[0], Distance(maximum_pos[0], "microns"), movementType)
        stage.goto(axes[1], Distance(maximum_pos[1], "microns"), movementType)
        stage.goto(focus_axis, maximum_plane, movementType)
        log.info(f"Moved to maximum of {maximum} at ({focus_axis}, {axes[0]}, {axes[1]}) = " +
                f"({maximum_plane.microns}, {maximum_pos[0]}, {maximum_pos[1]})")
        return

    #   real fitting for the 3d best point, either:
    #   measure points along the best fit line to the maxima
    #   find best fit to the beam waist, given planar widths
    #   true 3d fit to beam shape (requires grid_search to spit back more)
