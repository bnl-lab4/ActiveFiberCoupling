import lmfit
import sigfig
import numpy as np
import matplotlib.pyplot as plt
from MovementClasses import Distance

VALID_AXES = {'x', 'y', 'z'}


def val_unc(param: lmfit.parameter.Parameter):
    if param.stderr == 0:
        return sigfig.round(str(param.value), 3) + '+/- ___ '
    return sigfig.round(param.value, uncertainty=param.stderr)


def plane_fit_string(fit_result, axes):
    best_fit = []
    best_fit.append(f"A = {val_unc(fit_result.params['height'])}")
    best_fit.append(r'$\sigma_r$' + f" = {val_unc(fit_result.params['sigmax'])}" + r'$\mathrm{\mu m}$')
    best_fit.append(f"{axes[0].upper()} = {val_unc(fit_result.params['centerx'])}" + r'$\mathrm{\mu m}$')
    best_fit.append(f"{axes[1].upper()} = {val_unc(fit_result.params['centery'])}" + r'$\mathrm{\mu m}$')
    best_fit.append(r'C' + f" = {val_unc(fit_result.params['c'])}")
    return '\n'.join(best_fit)


def para_fit_string(fit_result, axis):
    best_fit = []
    best_fit.append(f"$w({axis}) = a x^2 + b x + c$")
    best_fit.append("$a$" + f" = {val_unc(fit_result.params['a'])}" + r"$\mathrm{\mu m^{-1}}$")
    best_fit.append("$b$" + f" = {val_unc(fit_result.params['b'])}")
    best_fit.append("$c$" + f" = {val_unc(fit_result.params['c'])}" + r"$\mathrm{\mu m}$")
    return '\n'.join(best_fit)


def lin_fit_string(fit_result, axes):
    best_fit = []
    best_fit.append(f"$ {axes[0]} = m {axes[1]} + b $")
    best_fit.append("$m$" + f" = {val_unc(fit_result.params['slope'])}" + r"$\mathrm{\mu m}$")
    best_fit.append("$b$" + f" = {val_unc(fit_result.params['intercept'])}" + r"$\mathrm{\mu m}$")
    return '\n'.join(best_fit)


def Gbeam_fit_string(fit_result, axes):
    focus_axis = list(VALID_AXES.difference(set(axes)))[0]
    best_fit = []
    best_fit.append(f"$I_0$ = {val_unc(fit_result.params['I0'])}")
    best_fit.append(f"$w_0$ = {val_unc(fit_result.params['w0'])}" + r"$\mathrm{\mu m}$")
    best_fit.append(f"$C$ = {val_unc(fit_result.params['C'])}")
    best_fit.append("$" + axes[0] + r"_\mathrm{waist}$" +
            f" = {val_unc(fit_result.params['waistx1'])}" + r"$\mathrm{\mu m}$")
    best_fit.append("$" + axes[1] + r"_\mathrm{waist}$" +
            f" = {val_unc(fit_result.params['waistx2'])}" + r"$\mathrm{\mu m}$")
    best_fit.append("$" + focus_axis + r"_\mathrm{waist}$" +
            f" = {val_unc(fit_result.params['waistx3'])}" + r"$\mathrm{\mu m}$")
    best_fit = ', '.join(best_fit[:3]) + '\n' + ', '.join(best_fit[3:])
    return best_fit


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
    fig.colorbar(resid_c, cax=axs[4], label='Residual (Absolute)')

    titles = ('Data', '', 'Best Fit', 'Data - Best Fit')
    for ax, title in zip(axs, titles):
        if ax == axs[1]:
            continue
        ax.set_xlabel(axes[0] + ' (microns)')
        ax.set_ylabel(axes[1] + ' (microns)')
        ax.set_title(title)

    axs[2].sharey(axs[0])
    axs[2].tick_params(labelleft=False)
    axs[3].sharey(axs[0])
    axs[3].tick_params(labelleft=False)

    fig.suptitle(f"{focus_axis.upper()} = {plane.prettyprint()}")
    axs[2].annotate(plane_fit_string(fit_result, axes), (0.0, 0.0), xytext=(0.01, 0.72), xycoords='axes fraction',
                        annotation_clip=True, fontsize=10, color='white')

    return fig


def plot_plane(response_grid: np.array, axis0_grid: np.array, axis1_grid: np.array,
               axes: list, plane: Distance):
    focus_axis = list(VALID_AXES.difference(set(axes)))[0]
    fig, data_ax = plt.subplots(layout = 'constrained')
    # pcolormesh correctly handles the cell centering for the given x and y arrays
    c = data_ax.pcolormesh(axis0_grid, axis1_grid, response_grid, shading='auto')
    fig.colorbar(c, ax=data_ax, label='Intensity')

    data_ax.set_xlabel(axes[0] + ' (microns)')
    data_ax.set_ylabel(axes[1] + ' (microns)')
    fig.suptitle(f"{focus_axis.upper()} = {plane.prettyprint()}", fontsize=10)

    return fig


def plot_para_fit(axes: str, waists: np.ndarray, waists_unc: np.ndarray, planes_microns: np.ndarray,
                  result: lmfit.model.ModelResult, show_plot: bool = False, log_plot: bool = True):

    fake_unc = all(waists_unc == 1)
    focus_axis = list(VALID_AXES.difference(set(axes)))[0]
    planes_range = planes_microns.max() - planes_microns.min()
    ext_factor = 0.1
    dense_lims = (planes_microns.min() - ext_factor * planes_range,
                  planes_microns.max() + ext_factor * planes_range)
    planes_dense = np.linspace(*dense_lims, 1000)
    waists_dense = result.eval(x=planes_dense)
    fit_waists = result.eval(x=planes_microns)
    resid = waists - fit_waists

    fig, axs = plt.subplots(nrows=2, figsize=(6, 6), layout='constrained', sharex=True,
                            gridspec_kw = dict(height_ratios = (1, 0.3)))
    axs[0].scatter(planes_microns, waists)
    if not fake_unc:
        axs[0].errorbar(planes_microns, waists, yerr=waists_unc, fmt='none')
    axs[0].plot(planes_dense, waists_dense, color='C1', label=para_fit_string(result, focus_axis))
    axs[1].scatter(planes_microns, resid)
    if not fake_unc:
        axs[1].errorbar(planes_microns, resid, yerr=waists_unc, fmt='none')

    axs[1].axhline(0, color='black', alpha=0.3)

    axs[0].legend()
    axs[0].grid(axis='both', which='both')
    axs[1].grid(axis='both', which='both')

    axs[1].set_xlabel(f"{focus_axis} (microns)")
    axs[0].set_ylabel(f"Width w({focus_axis}) (microns)")
    axs[1].set_ylabel("Residuals (Absolute)")

    return fig


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
        axrow[2].tick_params(labelleft=False)
        axrow[3].sharey(axrow[0])
        axrow[3].tick_params(labelleft=False)

        axrow[0].set_title("Data")
        axrow[2].set_title(f"Fit\nPlane {focus_axis} = {plane:.0f} microns")
        axrow[3].set_title('Data - Fit')

        axrow[0].set_ylabel(axes[1] + ' (microns)')
        for ax in (axrow[0], axrow[2], axrow[3]):
            ax.set_xlabel(axes[0] + ' (microns)')

    for ax in axs[:-1, 0]:
        ax.sharex(axs[-1, 0])
    for ax in axs[:-1, 2]:
        ax.sharex(axs[-1, 2])
    for ax in axs[:-1, 3]:
        ax.sharex(axs[-1, 3])

    fig.suptitle(Gbeam_fit_string(result, axes))

    return fig


def plot_lin_fit(axis: str, focus_axis: str, axis_peak_pos: np.ndarray,
                 axis_peak_unc: np.ndarray, planes_microns: np.ndarray, result: lmfit.model.ModelResult):
    fake_unc = all(axis_peak_unc == 1)
    planes_range = planes_microns.max() - planes_microns.min()
    ext_factor = 0.1
    dense_lims = (planes_microns.min() - ext_factor * planes_range,
                  planes_microns.max() + ext_factor * planes_range)
    planes_dense = np.linspace(*dense_lims, 100)
    axis_peak_dense = result.eval(x=planes_dense)
    fit_axis_peak = result.eval(x=planes_microns)
    resid = axis_peak_pos - fit_axis_peak

    fig, axs = plt.subplots(nrows=2, figsize=(6, 6), layout='constrained', sharex=True,
                            gridspec_kw = dict(height_ratios = (1, 0.3)))
    axs[0].scatter(planes_microns, axis_peak_pos)
    if not fake_unc:
        axs[0].errorbar(planes_microns, axis_peak_pos, yerr=axis_peak_unc, fmt='none')
    axs[0].plot(planes_dense, axis_peak_dense, color='C1', label=lin_fit_string(result, axis + focus_axis))
    axs[1].scatter(planes_microns, resid)
    if not fake_unc:
        axs[1].errorbar(planes_microns, resid, yerr=axis_peak_unc, fmt='none')

    axs[1].axhline(0, color='black', alpha=0.3)

    axs[0].legend()
    axs[0].grid(axis='both', which='both')
    axs[1].grid(axis='both', which='both')

    axs[1].set_xlabel(f"{focus_axis} (microns)")
    axs[0].set_ylabel(axis + ' (microns)')
    axs[1].set_ylabel("Residuals (Absolute)")

    return fig
