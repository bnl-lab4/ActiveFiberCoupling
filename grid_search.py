########### TODO:
# more testing
# deviation angles in 3d fit
# spiral order of grid locations
###########

import logging
import numpy as np
import lmfit
import copy
import matplotlib.pyplot as plt
from collections.abc import Sequence as sequence
from typing import Optional, Union, Tuple, Sequence
from datetime import datetime

from MovementClasses import MovementType, StageDevices
from Distance import Distance

# unique logger name for this module
log = logging.getLogger(__name__)

# max numpy arr size before it summarizes when printed
np.set_printoptions(threshold = 2000, linewidth=250)

VALID_AXES = {'x', 'y', 'z'}
WAVELENGTH = 0.65   # microns


def gaussbeam(x1, x2, x3, waistx1=0.0, waistx2=0.0, waistx3=0.0, I0=1.0, w0=8*np.pi*WAVELENGTH, C=0.0):
    pR2 = (np.pi * w0**2 / 0.65)**2     # squared Rayleigh range in the propagation direction

    r2 = (x1 - waistx1)**2 + (x2 - waistx2)**2  # squared distance from propagation axis
    p2 = (x3 - waistx3)**2          # squared position along propagation axis
    wp2 = w0**2 * (1 + p2 / pR2)    # squared waist at the given x3 position

    return I0 * (w0**2 / wp2) * np.exp(-2 * r2 / wp2) + C


def Gbeamfit_3d(axes: str, axis0_cube: np.ndarray, axis1_cube: np.ndarray,
                focus_cube: np.ndarray, data_cube: np.ndarray,
                show_plot: bool = False, log_plot: bool = True):

    assert axis0_cube.shape == axis1_cube.shape == focus_cube.shape == data_cube.shape, \
            "All position/data inputs must have the same shape"

    model = lmfit.Model(gaussbeam, independent_vars=['x1', 'x2', 'x3'])
    params = model.make_params()

    # guess initial values
    params['I0'].value = data_cube.max()
    max_idx = np.unravel_index(data_cube.argmax(), data_cube.shape)
    params['waistx1'].value = axis0_cube[max_idx]
    params['waistx2'].value = axis1_cube[max_idx]
    params['waistx3'].value = focus_cube[max_idx]

    weights = data_cube
    weights = weights - weights.min() + 1e-6 * np.ones_like(weights)
    weights /= weights.max()
    weights = np.sqrt(weights)

    position_cubes = dict(x1=axis0_cube.flatten(), x2=axis1_cube.flatten(), x3=focus_cube.flatten())
    result = model.fit(data=data_cube.flatten(), **position_cubes, params=params, weights=weights.flatten())
    log.debug("Fit complete")

    # plot logic
    fig = plot_3dfit(axes, *[cube.reshape(focus_cube.shape) for cube in position_cubes.values()], result)
    log.info("Plot Generated")
    if log_plot:
        fig.savefig(f"./log_plots/{str(datetime.now())[:-7].replace(' ', '_')}_" +
            "Gaussian_beam_fit.png",
                    format='png', facecolor='white', dpi=200)
    if show_plot:
        plt.show(block=True)
    plt.close()

    return result


def Gbeamfit_2d(axes: str, axis0_grid: np.ndarray,
                axis1_grid: np.ndarray, grid_values: np.ndarray,
                plane: Distance, show_plot: bool = False, log_plot: bool = True):

    gmodel = lmfit.models.Gaussian2dModel()
    params = gmodel.guess(grid_values.flatten(), axis0_grid.flatten(), axis1_grid.flatten())
    model = gmodel + lmfit.models.ConstantModel()
    params.add('c', value=grid_values.mean())
    params['sigmay'].expr('sigmax')     # circular gaussian

    weights = grid_values
    weights = weights - weights.min() + 1e-6 * np.ones_like(weights)
    weights /= weights.max()
    weights = np.sqrt(weights)

    result = model.fit(grid_values.flatten(), x=axis0_grid.flatten(), y=axis1_grid.flatten(),
                       params=params, weights=weights)
    # logic
    plot_2dfit()

    return result


def plot_3dfit(axes: str, axis0_cube: np.ndarray, axis1_cube: np.ndarray, focus_cube: np.ndarray,
               result: lmfit.model.ModelResult):
    focus_axis = list(VALID_AXES.difference(set(axes)))[0]
    data_cube = result.data.reshape(axis0_cube.shape)

    axis0_dense = np.arange(axis0_cube.min(), axis0_cube.max() + 1e-3, 1)
    axis1_dense = np.arange(axis1_cube.min(), axis1_cube.max() + 1e-3, 1)
    axis0_grid_dense, axis1_grid_dense = np.meshgrid(axis0_dense, axis1_dense)

    fig, axs = plt.subplots(nrows=len(focus_cube), ncols=5, figsize=((10, 3*len(focus_cube)+1)),
                            layout='constrained', gridspec_kw = dict(width_ratios = (1, 0.05, 1, 1, 0.05)))
    for axrow, grid_values, focus_grid in zip(axs, data_cube, focus_cube):
        plane = focus_grid[0, 0]
        focus_grid_dense = plane * np.ones_like(axis0_grid_dense)
        fit_grid_dense = result.eval(x1=axis0_grid_dense, x2=axis1_grid_dense, x3=focus_grid_dense)
        fit_grid = result.eval(x1=axis0_cube[0], x2=axis1_cube[0], x3=focus_grid)
        resid_grid = grid_values - fit_grid

        vmin = min(grid_values.min(), fit_grid_dense.min())
        vmax = max(grid_values.max(), fit_grid_dense.max())

        data_cbar = axrow[0].pcolormesh(axis0_cube[0], axis1_cube[0], grid_values,
                            vmin=vmin, vmax=vmax, shading='auto')
        axrow[2].pcolormesh(axis0_grid_dense, axis1_grid_dense, fit_grid_dense,
                            vmin=vmin, vmax=vmax, shading='auto')
        resid_cbar = axrow[3].pcolormesh(axis0_cube[0], axis1_cube[0], resid_grid, shading='auto')

        fig.colorbar(data_cbar, cax=axrow[1], label='Intensity')
        fig.colorbar(resid_cbar, cax=axrow[4], label='Residual (Absolute)')

        axrow[2].sharey(axrow[0])
        axrow[2].set_yticklabels([])
        axrow[3].sharey(axrow[0])
        axrow[3].set_yticklabels([])

        axrow[0].set_title("Data")
        axrow[2].set_title(f"Fit\nPlane {focus_axis} = {plane:.0f} microns")
        axrow[3].set_title('Data - Fit')

        axrow[0].set_ylabel(axes[1] + ' (microns)')
        for ax in (axrow[0], axrow[2], axrow[3]):
            ax.set_xlabel(axes[0] + ' (microns)')

    for ax in axs[:-1, 0]:
        ax.sharex(axs[-1, 0])
        ax.set_xticklabels([])
    for ax in axs[:-1, 2]:
        ax.sharex(axs[-1, 2])
        ax.set_xticklabels([])
    for ax in axs[:-1, 3]:
        ax.sharex(axs[-1, 3])
        ax.set_xticklabels([])

    return fig


def plane_fit_string(fit_result, axes):
    param_dict = {axes[0] : 'centerx', axes[1] : 'centery',
                  r'$\sigma_r$' : 'sigmax'}
    best_fit = [f"A = {fit_result.params['height'].value:.2f}", ]
    best_fit += [f"{key} = {fit_result.params[name].value:.0f}" for key, name in param_dict.items()]
    return ', '.join(best_fit)


def plot_plane(response_grid: np.array, axis0_grid: np.array, axis1_grid: np.array,
               axes: list, plane: Distance):
    focus_axis = list(VALID_AXES.difference(set(axes)))[0]
    fig, data_ax = plt.subplots(layout = 'constrained')
    # pcolormesh correctly handles the cell centering for the given x and y arrays
    c = data_ax.pcolormesh(axis0_grid, axis1_grid, response_grid, shading='auto')
    fig.colorbar(c, ax=data_ax, label='Intensity')

    data_ax.set_xlabel(axes[0] + ' (microns)')
    data_ax.set_ylabel(axes[1] + ' (microns)')
    data_ax.set_title('Data')
    fig.suptitle(f"{focus_axis.upper()} = {plane.prettyprint()}")

    return fig


def plot_2dfit(response_grid: np.array, axis0_grid: np.array, axis1_grid: np.array,
               axes: list, plane: Distance, fit_result: lmfit.model.ModelResult):
    focus_axis = list(VALID_AXES.difference(set(axes)))[0]
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

    fig.suptitle(f"{focus_axis.upper()} = {plane.prettyprint()}")
    axs[2].annotate(plane_fit_string(fit_result, axes), (0.5, 0.95), xytext=(0.1, 0.95), xycoords='axes fraction',
                        annotation_clip=True, fontsize=10, color='white')

    return fig


def plane_grid(stage: StageDevices, movementType: MovementType, plane: Distance,
               axes: list, axis0: np.array, axis1: np.array,
               exposureTime: Union[int, float],
               show_plot: bool = False, log_plot: bool = True):

    focus_axis = list(VALID_AXES.difference(set(axes)))[0]

    axis0_grid, axis1_grid = np.meshgrid(axis0, axis1)
    response_grid = np.zeros_like(axis0_grid)
    for i, pos0 in enumerate(axis0):
        stage.goto(axes[0], Distance(pos0, "microns"), movementType)
        for j, pos1 in enumerate(axis1):
            stage.goto(axes[1], Distance(pos1, "microns"), movementType)
            response_grid[j, i] += stage.integrate(exposureTime)

    log.info(f"Grid values:\n{response_grid}")

    fig = plot_plane(response_grid, axis0_grid, axis1_grid, axes, plane)
    log.info("Plot Generated")
    if log_plot:
        fig.savefig(f"./log_plots/{str(datetime.now())[:-7].replace(' ', '_')}_" +
            f"{movementType.value}_{focus_axis}-{plane.prettyprint()}.png",
                    format='png', facecolor='white', dpi=200)
    if show_plot:
        plt.show(block=True)
    plt.close()

    return response_grid


def run(stage: StageDevices, movementType: MovementType, exposureTime: Union[int, float],
        spacing: Union[None, Distance, Sequence[Distance]] = None,  # default 10 volts
        num_points: Tuple[None, int, Sequence[int]] = None,
        center: Union[None, str, Sequence[Distance]] = None,
        limits: Optional[Sequence[Distance]] = None, # 2-list of (2-lists of) Distance objects
        axes: str = 'xz', planes: Union[None, int, Sequence[Distance]] = None,   # default 3 planes
        fit_planes: bool = True, fit_3d: bool = True,
        grid_kwargs: dict = {}, fit_3d_kwargs: dict = {}, fit_2d_kwargs: dict = {}):

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
    if isinstance(limits, sequence) and all(isinstance(limit, Distance) for limit in limits):
        limits = [copy.deepcopy(limits) for i in range(2)]
    if isinstance(spacing, Distance):
        spacing = (spacing, spacing)
    if isinstance(num_points, int):
        num_points = (num_points, num_points)

    if isinstance(planes, int):
        if movementType == MovementType.PIEZO:
            planes_limits = [limit.microns for limit in stage.axes[focus_axis].PIEZO_LIMITS]
        elif movementType == MovementType.STEPPER:
            planes_limits = [limit.microns for limit in stage.axes[focus_axis].STEPPER_LIMITS]
        planes = np.linspace(*planes_limits, planes)
        planes = [Distance(plane, "microns") for plane in planes]

    if center is not None:
        limits[0][0] += center[0]
        limits[0][1] += center[0]
        limits[1][0] += center[1]
        limits[1][1] += center[1]

    if spacing is not None:
        axis0 = np.arange(limits[0][0].microns, (1+1e-6)*limits[0][1].microns, spacing[0].microns)
        axis1 = np.arange(limits[1][0].microns, (1+1e-6)*limits[1][1].microns, spacing[1].microns)
        # 1+1e-6 is so that the upper limit is included in the resulting list
    if num_points is not None:
        axis0 = np.linspace(limits[0][0].microns, limits[0][1].microns, num_points[0])
        axis1 = np.linspace(limits[1][0].microns, limits[1][1].microns, num_points[1])

    axis0_grid, axis1_grid = np.meshgrid(axis0, axis1)

    # take data in planes
    grid_values = []
    for n, plane in enumerate(planes):
        log.info(f"Running grid search in plane {focus_axis} = " +
                    f"{plane.prettyprint()} microns with grid:\n" +
                    f"{axes[0]} = {axis0} microns\n{axes[1]} = {axis1} microns")
        stage.goto(focus_axis, plane, movementType)

        plane_values = plane_grid(stage, movementType, plane,
                                    axes, axis0, axis1, exposureTime, **grid_kwargs)
        grid_values.append(plane_values)

    # make cubes
    grid_values = np.array(grid_values)
    axis0_cube = np.tile(axis0_grid, (len(planes), 1, 1))
    axis1_cube = np.tile(axis1_grid, (len(planes), 1, 1))
    focus_cube = np.array([plane.microns * np.ones_like(axis0_grid) for plane in planes])

    if fit_3d:
        if len(planes) <= 1:
            log.warning("There are not enough planes to do a 3d Gaussian beam model fit")
        else:
            log.info("Attempting to fit data with a 3d Gaussian beam model")

            result = Gbeamfit_3d(axes, axis0_cube, axis1_cube, focus_cube, grid_values, **fit_3d_kwargs)
            if result.success:

                peak = result.params['I0'].value
                best_pos = [Distance(result.params[f"waistx{n}"].value, 'microns') for n in range(1, 4)]
                stage.goto(axes[0], best_pos[0], movementType)
                stage.goto(axes[1], best_pos[1], movementType)
                stage.goto(focus_axis, best_pos[2], movementType)

                log.info(f"Moved to maximum of {peak} at ({axes[0]}, {axes[1]}, {focus_axis}) = " +
                        f"({best_pos[0].prettyprint()}, {best_pos[1].prettyprint()}," +
                        f"{best_pos[2].prettyprint()})")
                return
            log.info("3d Gaussian beam fit failed")

    if fit_planes:
        log.info("Attempting to fit planes with Gaussian models")
        plane_results = []
        for plane in planes:
            result = Gbeamfit_2d(axes, axis0_grid, axis1_grid, grid_values,
                                                     plane, **fit_2d_kwargs)
            plane_results.append(result)
            log.info("plane fit information")

        successful = []
        successful_planes = []
        for plane, result in zip(planes, plane_results):
            if result.success:
                successful.append(result)
                successful_planes.append(plane)

        log.info(f"{len(successful)} of {len(planes)} planes successfully fit to")

        if len(successful) >= 3:
            log.info("Fitting parabola to waists")
            waists = np.array([result.params['sigmax'].value for result in successful])
            waists_unc = np.array([result.params['sigmax'].stderr for result in successful])

            # fit parabola to waists
            model = lmfit.models.QuadraticModel()
            params = model.guess(waists, x=successful_planes)
            para_result = model.fit(waists, x=successful_planes, params=params, weights=1/waists_unc)
            focus_pos = - para_result.params['b'].value / 2 * para_result.params['a'].value
            waist_min = Distance(para_result.eval(focus_pos), 'microns')
            waist_pos = [None, None, Distance(focus_pos, 'microns')]

            if para_result.success:
                log.info("Fitting linear functions to peak center values")
                # fit peak position values (in case the beam is not parallel with the focal axis)
                axis0_peak_pos = np.array([result.params['centerx'].value for result in successful])
                axis0_peak_unc = np.array([result.params['centerx'].stderr for result in successful])
                model = lmfit.models.LinearModel()
                params = model.guess(axis0_peak_pos, x=successful_planes)
                lin_result0 = model.fit(axis0_peak_pos, x=successful_planes,
                                       params=params, weights=1/axis0_peak_unc)

                # Take mean value if linear fit fails
                if lin_result0.success:
                    waist_pos[0] = Distance(lin_result0.eval(waist_pos[2].microns), 'microns')
                else:
                    waist_pos[0] = Distance(axis0_peak_pos.mean(), 'microns')
                    log.info(f"{axes[0]} linear fit failed")

                axis1_peak_pos = np.array([result.params['centery'].value for result in successful])
                axis1_peak_unc = np.array([result.params['centery'].stderr for result in successful])
                model = lmfit.models.LinearModel()
                params = model.guess(axis1_peak_pos, x=successful_planes)
                lin_result1 = model.fit(axis1_peak_pos, x=successful_planes,
                                       params=params, weights=1/axis1_peak_unc)

                if lin_result1.success:
                    waist_pos[1] = Distance(lin_result1.eval(waist_pos[2].microns), 'microns')
                else:
                    waist_pos[1] = Distance(axis0_peak_pos.mean(), 'microns')
                    log.info(f"{axes[1]} linear fit failed")

                stage.goto(axes[0], waist_pos[0], movementType)
                stage.goto(axes[1], waist_pos[1], movementType)
                stage.goto(focus_axis, waist_pos[2], movementType)

                log.info(f"Moved to minimum waist of {waist_min} at ({axes[0]}, {axes[1]}, {focus_axis}) = " +
                        f"({waist_pos[0].prettyprint()}, {waist_pos[1].prettyprint()}," +
                        f"{waist_pos[2].prettyprint()})")
                return

        # if any fits were successful, go to the highest known best-fit peak
        if len(successful) != 0:
            plane_peaks = np.array([result.params['height'].value for result in successful])
            max_peak = plane_peaks.max()
            max_peak_idx = np.unravel_index(plane_peaks.argmax(), plane_peaks.shape)

            best_plane = successful[max_peak_idx]
            best_pos = [Distance(best_plane.params['centerx'].value, 'microns'),
                        Distance(best_plane.params['centery'].value, 'microns'),
                        successful_planes[max_peak_idx]]

            stage.goto(axes[0], best_pos[0], movementType)
            stage.goto(axes[1], best_pos[1], movementType)
            stage.goto(focus_axis, best_pos[2], movementType)

            log.info(f"Moved to maximum of known peaks {max_peak} at ({axes[0]}, {axes[1]}, {focus_axis}) = " +
                    f"({best_pos[0].prettyprint()}, {best_pos[1].prettyprint()}," +
                    f"{best_pos[2].prettyprint()})")
            return

        log.info("No planes were successfully fit with a gaussian")

    # go to position of brightest position visited
    max_value = grid_values.max()
    max_value_idx = np.unravel_index(grid_values.argmax(), grid_values.shape)
    max_value_pos = [Distance(axis0_cube[max_value_idx], "microns"),
                     Distance(axis1_cube[max_value_idx], 'microns'),
                     Distance(focus_cube[max_value_idx], 'microns')]

    stage.goto(axes[0], max_value_pos[0], movementType)
    stage.goto(axes[1], max_value_pos[1], movementType)
    stage.goto(focus_axis, max_value_pos[2], movementType)

    log.info(f"Moved to maximum of visited positions {max_value} at ({axes[0]}, {axes[1]}, {focus_axis}) = " +
            f"({max_value_pos[0].prettyprint()}, {max_value_pos[1].prettyprint()}," +
            f"{max_value_pos[2].prettyprint()})")
    return
