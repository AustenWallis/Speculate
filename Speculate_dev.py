# %% Stage 1: Imports and Speculate Grid Classes/funcitons for 'python'.

# 1) Imports
import autopep8
import multiprocessing as mp
import os
import random
import emcee
import corner
# import fnmatch
import numpy as np
import math as m
import matplotlib.pyplot as plt
import arviz as az
import scipy.stats as st
import Speculate_addons.Spec_functions as spec

from multiprocessing import Pool
from multiprocessing import cpu_count
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from Starfish.grid_tools import HDF5Creator
from Starfish.grid_tools import GridInterface
from Starfish.emulator import Emulator
from Starfish.emulator.plotting import plot_eigenspectra
from Starfish.spectrum import Spectrum
from Starfish.models import SpectrumModel
from Speculate_addons.Spec_gridinterfaces import KWDGridInterface
from Speculate_addons.Spec_gridinterfaces import ShortSpecGridInterface

# %% Stage 2: Setting up and processing the flux grid space (HDF5).

# 2) Flux grid space setup and PCA inputs
""" 
-------------------------------------------------------------------------------|
                          - Set up your grid space! -
        Parameters of model            |            Parameters of model
        (kgrid/20years old)            |         (short spec cv grid/v87a)
-------------------------------------------------------------------------------|
1) wind.mdot (msol/yr)                 | 1) wind.mdot (msol/yr)
2) kn.d                                | 2) KWD.d
3) kn.mdot_r_exponent                  | 3) KWD.v_infinity (in_units_of_vescape)
4) kn.v_infinity (in_units_of_vescape) |
5) kn.acceleration_length (cm)         |
6) kn.acceleration_exponent            |
max_wl_range = (876, 1824)             | max_wl_range = (850, 1850)
-------------------------------------------------------------------------------|
"""

# ----- Inputs here -----------------------------------------------------------|
model_parameters = (1, 2, 3)  # Minimum 2 parameter tuple from table above
wl_range = (1510, 1590)       # Wavelength range of your emulator grid space.
                              # Later becomes truncated +/-10Angstom
                              
scale = 'linear'              # Transformation scaling for flux data. 'linear'
                              # 'log' or 'scaled'. scale not implemented yet.

grid_file_name = 'Grid_full'  # Builds fast, file save unnessary.
kgrid = 0                     # Turn on if planning to use kgrid
shortspec = 1                 # Turn on if planning to use shortspec_cv_grid

n_components = 4              # Alter the number of PCA components used.
# Integer for no. of components or decimal (0.0-1.0) for 0%-100% accuracy.
# -----------------------------------------------------------------------------|

# Sorting parameters by increasing order
model_parameters = sorted(model_parameters)
# Looping through parameters to create a string of numbers for file name
model_parameters_str = ''.join(str(i) for i in model_parameters)

# Selecting the specified grid interface 
if kgrid == 1:
    grid = KWDGridInterface(
        path='kgrid/sscyg_kgrid090311.210901/',
        wl_range=wl_range,
        model_parameters=model_parameters)
    # Change inclination with usecols[1]
    usecols = (1, 8) # Wavelength, Inclination 8-14 --> 40-70 degrees
    skiprows = 2  # Start of data within file
    inclination = usecols[1] * 5
    emu_file_name = f'Kgrid_emu_{scale}_{usecols[1]*5}inc_{wl_range[0]}-{wl_range[1]}AA_{n_components}comp_{model_parameters_str}'

if shortspec == 1:
    grid = ShortSpecGridInterface(
        path='short_spec_cv_grid/',
        wl_range=wl_range,
        model_parameters=model_parameters, 
        scale=scale
        )
    # Change inclination with usecols[1]
    usecols = (1, 16) # Wavelength, Inclination 10-21 --> 30-85 degrees
    skiprows = 81  # Starting point of data within file
    inclination = (usecols[1]-4) * 5
    emu_file_name = f'SSpec_emu_{scale}_{(usecols[1]-4) * 5}inc_{wl_range[0]}-{wl_range[1]}AA_{n_components}comp_{model_parameters_str}'

# Add custom name if auto generation undesired?
#emu_file_name = 'custom_name.hdf5' 

# Checking if the emulator file has been created
if os.path.isfile(f'Grid-Emulator_Files/{emu_file_name}.hdf5'):
    print(f'Emulator {emu_file_name} already exists.')
    emu = Emulator.load(f"Grid-Emulator_Files/{emu_file_name}.hdf5")
    emu_exists = 1
    print('Existing emulator loaded.')
else:
    print(f'Emulator {emu_file_name} does not exist.')
    emu_exists = 0
    print('Create new emulator in Stage 3.')
    
# Faster processing with python's .spec files into a hdf5 file
# Auto-generated keyname's required to integrate with Starfish
keyname = ["param{}{{}}".format(i) for i in model_parameters]
keyname = ''.join(keyname)
# Processing to HDF5 file interface
creator = HDF5Creator(
    grid,
    f'Grid-Emulator_Files/{grid_file_name}.hdf5',
    key_name=keyname,
    wl_range=wl_range)
creator.process_grid()


# %% Stage 3: Generating/training emulator

# 3) Generating and training a new emulator
# Asking if user wants to continue training a new emulator
if emu_exists == 1:
    print("Emulator's name:", emu_file_name)
    print('Do you want to overwrite the existing emulator (y/n)?')
    if input('y/n: ') == 'y':
        emu_exists = 0
        print('Existing emulator will be overwritten')
    else:
        print('Existing emulator will be used')

# Generating/training/saving and displaying the new emulator
if emu_exists == 0:
    emu = Emulator.from_grid(
        f'Grid-Emulator_Files/{grid_file_name}.hdf5',
        n_components=n_components,
        svd_solver="full") 
    # scipy.optimise.minimise routine
    emu.train(method="Nelder-Mead", options=dict(maxiter=1e5, disp=True))
    emu.save(f'Grid-Emulator_Files/{emu_file_name}.hdf5')
    print(emu)  # Displays the trained emulator's parameters

# %% Stage 3 not converged?: Continue training emulator

# 3..) Continue training emulator
emu.train(method="Nelder-Mead", options=dict(maxiter=1e5, disp=True))
emu.save(f'Grid-Emulator_Files/{emu_file_name}.hdf5')  # Saving the emulator
print(emu)

# %% Stage 4: Plotting the emulator's eigenspectra and weights slice

# 4) Plotting emulator - TODO: Make better plots
# Inputs: Displayed parameter (1-X), other parameters' fixed index (0-(X-1))
spec.plot_emulator(emu, grid, model_parameters, 1, 0)
# plot_new_eigenspectra(emu, 51)  # <---- Yet to implement

# %% plot_new_eigenspectra function


def plot_new_eigenspectra(emulator, params, filename=None):
    from matplotlib import gridspec
    """

    Parameters
    ----------
    emulator
    params
    filename : str or path-like, optional
        If provided, will save the plot at the given filename

    Example of a deconstructed set of eigenspectra

    .. figure:: assets/eigenspectra.png
        :align: center
    """
    weights = emulator.weights[params]
    X = emulator.eigenspectra * emulator.flux_std
    reconstructed = weights @ emulator.eigenspectra + emulator.flux_mean
    reconstructed *= emulator.norm_factor(emulator.grid_points[params])
    reconstructed = np.squeeze(reconstructed)
    height = int(emulator.ncomps) * 1.25
    fig = plt.figure(figsize=(8, height))
    gs = gridspec.GridSpec(
        int(emulator.ncomps) + 1,
        1,
        height_ratios=[3] + list(np.ones(int(emulator.ncomps))),
    )
    ax = plt.subplot(gs[0])
    ax.plot(emulator.wl, reconstructed, lw=1)
    ax.set_ylabel("$f_\\lambda$ [erg/cm^2/s/A]")
    plt.setp(ax.get_xticklabels(), visible=False)
    for i in range(emulator.ncomps):
        ax = plt.subplot(gs[i + 1], sharex=ax)
        ax.plot(emulator.wl, emulator.eigenspectra[i], c="0.4", lw=1)
        ax.set_ylabel(rf"$\xi_{i}$")
        if i < emulator.ncomps - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
        ax.legend([rf"$w_{i}$ = {weights[i]:.2e}"])
    plt.xlabel("Wavelength (A)")
    plt.tight_layout(h_pad=0.2)

    plt.show()
# %% Stage 5: Emulator covariance matrix


emu = Emulator.load(f"Grid-Emulator_Files/{emu_file_name}.hdf5")
random_grid_point = random.choice(emu.grid_points)
print("Random Grid Point Selection")
print(list(emu.param_names))
print(emu.grid_points[0])
print(random_grid_point)
weights, cov = emu(emu.grid_points[0])  # or put: random_grid_point
X = emu.eigenspectra * (emu.flux_std)
flux = (weights @ X) + emu.flux_mean
emu_cov = X.T @ cov @ X
plt.matshow(emu_cov, cmap='Reds')
plt.title("Emulator Covariance Matrix")
plt.colorbar()
plt.show()
plt.plot(emu.wl, flux)

# %% DEBUG 1: Emulator against grid point correlation coefficient matrix

correlation_matrix = np.empty((len(emu.grid_points), len(
    emu.grid_points)))  # empty matrix of grid dimensions
# Iterating across the spectral file grid points (changing file names) (y-axis)
for i in tqdm(range(len(emu.grid_points))):
    # Loading and data manipulation performed on the grid files
    spectrum_file = grid.get_flux(emu.grid_points[i])
    waves, obs_flux_data = np.loadtxt(
        spectrum_file, usecols=usecols, unpack=True, skiprows=skiprows)
    wl_range_data = (wl_range[0] + 10, wl_range[1] - 10)
    waves = np.flip(waves)
    obs_flux_data = np.flip(obs_flux_data)
    # obs_flux_data = gaussian_filter1d(obs_flux_data, 50)
    if scale == 'log':
        obs_flux_data = [np.log10(i) for i in obs_flux_data]  # logged scale
        obs_flux_data = np.array(obs_flux_data)
    if scale == 'scaled':
        obs_flux_data /= np.mean(obs_flux_data)
    indexes = np.where(
        (waves >= wl_range_data[0]) & (
            waves <= wl_range_data[1]))
    waves = waves[indexes[0]]
    obs_flux_data = obs_flux_data[indexes[0]]
    sigmas = np.zeros(len(waves))
    obs_data = Spectrum(waves, obs_flux_data, sigmas=sigmas, masks=None)
    # Loading and data manipulation performed on the emulation files utilising
    # starfish module.
    for j in range(len(emu.grid_points)
                   ):  # Iterating across the emulated files (x-axis)
        model = SpectrumModel(
            f'Grid-Emulator_Files/{emu_file_name}.hdf5',
            obs_data,
            grid_params=list(emu.grid_points[j]),  # [list, of , grid , points]
            Av=0,
            global_cov=dict(log_amp=-20, log_ls=5)  # Numbers irrelevant
        )
        model_flux = model.flux_non_normalised()
        a = np.corrcoef(obs_flux_data, model_flux)
        correlation_matrix[i][j] = a[0][1]

# Plotting the correlation coefficient matrix
plt.matshow(correlation_matrix)
plt.title(
    f"Correlation coefficient matrix of the emulated spectral grid point \n compared to the real spectral grid point \n {emu_file_name}")
plt.colorbar()
plt.xlabel('Emulator Grid Points')
plt.ylabel('Data Grid Points')
# plt.savefig('60inc_900-1800Araw_10comp_1234.png', dpi=300)
plt.plot()

# %% DEBUG 2: 2D emulator/grid point plotting

fig, axs = plt.subplots(3, 3)
interesting_1stparam = [4e-11, 6.6e-10, 3e-9]
# Assumed only 2 parameters in emu.grid_points
interesting_2ndparam = [1, 2, 3]
number_of_parameters = len(emu.param_names)
plot_row = 0
plot_column = 0
for i, w in enumerate(emu.grid_points):
    truth_array1 = np.isin(interesting_1stparam, w)
    truth_array2 = np.isin(interesting_2ndparam, w)
    truth_value1 = np.any(truth_array1)
    truth_value2 = np.any(truth_array2)
    if truth_value1 and truth_value2:
        # Loading and data manipulation performed on the grid files
        spectrum_file = grid.get_flux(emu.grid_points[i])
        waves, obs_flux_data = np.loadtxt(
            spectrum_file, usecols=usecols, unpack=True, skiprows=skiprows)
        wl_range_data = (wl_range[0] + 10, wl_range[1] - 10)
        waves = np.flip(waves)
        obs_flux_data = np.flip(obs_flux_data)
        # obs_flux_data = gaussian_filter1d(obs_flux_data, 50)
        if scale == 'log':
            obs_flux_data = [np.log10(i)
                             for i in obs_flux_data]  # logged scale
            obs_flux_data = np.array(obs_flux_data)
        if scale == 'scaled':
            obs_flux_data /= np.mean(obs_flux_data)
        indexes = np.where(
            (waves >= wl_range_data[0]) & (
                waves <= wl_range_data[1]))
        waves = waves[indexes[0]]
        obs_flux_data = obs_flux_data[indexes[0]]
        sigmas = np.zeros(len(waves))
        obs_data = Spectrum(waves, obs_flux_data, sigmas=sigmas, masks=None)
        # Loading and data manipulation performed on the emulation files
        # utilising starfish module.
        model = SpectrumModel(
            f'Grid-Emulator_Files/{emu_file_name}.hdf5',
            obs_data,
            grid_params=list(emu.grid_points[i]),  # [list, of , grid , points]
            Av=0,
            global_cov=dict(log_amp=-20, log_ls=5)  # Numbers irrelevant
        )
        model_flux = model.flux_non_normalised()
        a = np.corrcoef(obs_flux_data, model_flux)
        if plot_row == 3:
            plot_row = 0
            plot_column += 1
        if plot_row == 1 and plot_column == 1:
            fixed_obs_flux_data = obs_flux_data
            fixed_model_flux = model_flux
        axs[plot_row, plot_column].plot(
            waves, obs_flux_data, color='blue', label='Data', linewidth=0.5)
        axs[plot_row, plot_column].plot(
            waves, model_flux, color='green', label='Model', linewidth=0.5)
        axs[plot_row, plot_column].title.set_text(
            f'Corrcoef = {np.round(a[0][1],3)}')
        plot_row += 1

label1 = 0
label2 = 0
for ax in axs.flat:
    ax.set(
        xlabel=f'$\\lambda$ [$\\AA$] || $v_\\infty$ = {interesting_2ndparam[label2]}',
        ylabel='Flux || $\\dot{M}$' +
        f'={interesting_1stparam[label1]}')
    axs[label1,
        label2].plot(waves,
                     fixed_obs_flux_data,
                     color='red',
                     label='Fixed Middle Graph Data',
                     linewidth=0.5,
                     zorder=0,
                     alpha=0.5)
    label2 += 1
    if label2 == 3:
        label2 = 0
        label1 += 1
for ax in axs.flat:
    ax.label_outer()
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3)
fig.tight_layout()
fig.savefig('plots/2D Emu-Grid 3x3 plot.png', dpi=300)
fig.show()

# %% DEBUG 3:Individual spectrum plot
m_dot = [1e-10, 3e-9]
for i, w in enumerate([0, 24]):
    spectrum_file = grid.get_flux(emu.grid_points[w])
    waves, obs_flux_data = np.loadtxt(
        spectrum_file, usecols=usecols, unpack=True, skiprows=skiprows)
    wl_range_data = (wl_range[0] + 10, wl_range[1] - 10)
    waves = np.flip(waves)
    obs_flux_data = np.flip(obs_flux_data)
    # obs_flux_data = gaussian_filter1d(obs_flux_data, 50)
    if scale == 'log':
        obs_flux_data = [np.log10(i) for i in obs_flux_data]  # logged scale
        obs_flux_data = np.array(obs_flux_data)
    if scale == 'scaled':
        obs_flux_data /= np.mean(obs_flux_data)
    indexes = np.where(
        (waves >= wl_range_data[0]) & (
            waves <= wl_range_data[1]))
    waves = waves[indexes[0]]
    obs_flux_data = obs_flux_data[indexes[0]]
    sigmas = np.zeros(len(waves))
    obs_data = Spectrum(waves, obs_flux_data, sigmas=sigmas, masks=None)

    model = SpectrumModel(
        f'Grid-Emulator_Files/{emu_file_name}.hdf5',
        obs_data,
        grid_params=list(emu.grid_points[w]),  # [list, of, grid, points]
        Av=0,
        global_cov=dict(log_amp=-20, log_ls=5)  # Numbers irrelevant
    )
    model_flux = model.flux_non_normalised()

    plt.plot(waves, obs_flux_data, label=f'{m_dot[i]}')
# plt.plot(waves, model_flux, label='Model', color='green')
    plt.title(f'M_dot, v_inf = 1')
plt.legend()
plt.show()

# %% DEBUG 4: Spectrum Iteration plotting (Correlation coefficient Matrix)
# Spectrum Plotting !!!
correlation_matrix2 = np.empty((int(len(emu.grid_points) / 6),
                                int(len(emu.grid_points) / 6)))  # empty matrix of grid dimensions
plot_number = 1

# Iterating across the spectral file grid points (changing file names) (y-axis)
for i in tqdm(range(len(emu.grid_points))):
    # Loading and data manipulation performed on the grid files
    if i % 6 == 0:
        spectrum_file = grid.get_flux(emu.grid_points[i])
        waves, obs_flux_data = np.loadtxt(
            spectrum_file, usecols=usecols, unpack=True, skiprows=skiprows)
        wl_range_data = (wl_range[0] + 10, wl_range[1] - 10)
        waves = np.flip(waves)
        obs_flux_data = np.flip(obs_flux_data)
        # obs_flux_data = gaussian_filter1d(obs_flux_data, 50)
        if scale == 'log':
            obs_flux_data = [np.log10(i)
                             for i in obs_flux_data]  # logged scale
            obs_flux_data = np.array(obs_flux_data)
        if scale == 'scaled':
            obs_flux_data /= np.mean(obs_flux_data)
        indexes = np.where(
            (waves >= wl_range_data[0]) & (
                waves <= wl_range_data[1]))
        waves = waves[indexes[0]]
        obs_flux_data = obs_flux_data[indexes[0]]
        sigmas = np.zeros(len(waves))
        obs_data = Spectrum(waves, obs_flux_data, sigmas=sigmas, masks=None)
        # print(i, 'i')
        i = int(i / 9)
    # Loading and data manipulation performed on the emulation files utilising
    # starfish module.
        for j in range(len(emu.grid_points)
                       ):  # Iterating across the emulated files (x-axis)
            if j % 6 == 0:
                # print(j, 'j')
                model = SpectrumModel(
                    f'Grid-Emulator_Files/{emu_file_name}.hdf5',
                    obs_data,
                    # [list, of , grid , points]
                    grid_params=list(emu.grid_points[j]),
                    Av=0,
                    global_cov=dict(
                        log_amp=-20, log_ls=5)  # Numbers irrelevant
                )
                model_flux = model.flux_non_normalised()
                a = np.corrcoef(obs_flux_data, model_flux)
                j = int(j / 9)
                correlation_matrix2[i][j] = a[0][1]
                # plt.subplot(10,10,plot_number)
                j = int(j * 9)
                print(emu.grid_points[j])
                plt.plot(waves, obs_flux_data, label='data')
                plt.plot(waves, model_flux, label='model', color='green')
                plt.legend()
                plt.title(
                    f"{plot_number} across emulated grid points then down spectral grid points")
                plot_number += 1
                plt.show()

# plt.tight_layout(pad=5.0)
plt.show()
plt.savefig(f'{emu_file_name}_spectrums.png', dpi=300)
# Plotting the correlation coefficient matrix
plt.matshow(correlation_matrix2)
plt.title("Correlation coefficient matrix of the emulated spectral grid point \n compared to the real spectral grid point")
plt.colorbar()
plt.xlabel('Emulator Grid Points')
plt.ylabel('Data Grid Points')
plt.show()
parameter1 = [i[0] for i in emu.grid_points]
parameter2 = [i[1] for i in emu.grid_points]
# parameter3 = [i[2] for i in emu.grid_points]
mean1 = np.mean(parameter1)
mean2 = np.mean(parameter2)
# mean3 = np.mean(parameter3)
plt.title("Change in parameter values across grid points")
plt.plot(range(len(emu.grid_points)), parameter1 /
         mean1, label='Param1', color='red')
plt.plot(range(len(emu.grid_points)), parameter2 /
         mean2, label='Param2', color='green')
# plt.plot(range(len(emu.grid_points)), parameter3/mean3, label='Param3', color='blue')
# plt.label()
plt.show()


# %% Stage 6: Adding a spectrum data file as test data

# 6) Adding observational spectrum as data

# ---------- Switches here -----------|
# Four methods to select which type of testing spectrum file you want:
# Turn off/on (0/1) to use this method
data_one = 0                # [1] A grid point
data_two = 0                # [2] A noisy grid point
data_three = 0              # [3] Interpolatation between two grid points
data_four = 1               # [4] A custom test file from python
# ----------------------------------------------------------------------------|

# ----------- Inputs here ------------|
# 1)
if data_one == 1:
    # < File corresponds to grid points in section 2
    file = 'sscyg_k2_040102000001.spec'
    # < Parameter's point given by the XX number in the name (6 params = 12 digits)
    # < 040102000000 = 4.5e-10, 16, 1, 1, 1e10, 3
# 2)
if data_two == 1:
    file = 'sscyg_k2_040102000001.spec'  # < File naming same as 1)
    noise_std = 0.50                    # < Percentage noise (0.05 sigma)
# 3)
if data_three == 1:
    print('to do') # TODO
# 4)
if data_four == 1:
    # Python v87a formatting. 
    # Place python file within kgrid folder/directory
    #file = 'runtest_WMdot2e-10_d14_vinf1p5.spec'
    file = 'run59_WMdot4e-10_d5_vinf2p5.spec'
    
# ------------------------------------|

if data_one == 1:
    waves, fluxes = np.loadtxt(
        f'kgrid/sscyg_kgrid090311.210901/{file}', usecols=usecols, unpack=True, skiprows=skiprows)

if data_two == 1:
    waves, fluxes = np.loadtxt(
        f'kgrid/sscyg_kgrid090311.210901/{file}', usecols=usecols, unpack=True, skiprows=skiprows)
    noise = noise_std * np.std(fluxes)
    for i in range(len(waves)):
        fluxes[i] = np.random.normal(fluxes[i], noise)

if data_three == 1:
    print('to do')

if data_four == 1:
    waves, fluxes = np.loadtxt(
        f'observation_files/{file}', unpack=True, usecols=usecols, skiprows=skiprows)

# Data manipulation/truncation into correct format.
wl_range_data = (wl_range[0] + 10, wl_range[1] - 10)  # Truncation
waves = np.flip(waves)
fluxes = np.flip(fluxes)
# fluxes = gaussian_filter1d(fluxes, 50)
if scale == 'log':
    fluxes = [np.log10(i) for i in fluxes]  # log
    fluxes = np.array(fluxes)  # log
if scale == 'scaled':
    fluxes /= np.mean(fluxes)
indexes = np.where((waves >= wl_range_data[0]) & (
    waves <= wl_range_data[1]))  # Truncation + next 2 lines
waves = waves[indexes[0]]
fluxes = fluxes[indexes[0]]
raw_flux = list(fluxes)
sigmas = np.zeros(len(waves))
data = Spectrum(waves, fluxes, sigmas=sigmas, masks=None)
data.plot(yscale="linear")
print(file)


# %% DEBUG: Data spectrum data file and emulation model overplotting (normalisation)

# ----- Inputs here ------|
# Natural logarithm of the global covariance's Matern 3/2 kernel amplitude
# log=-52 'linear', log=-8 'log'
log_amp = -52
# Natural logarithm of the global covariance's Matern 3/2 kernel lengthscale
log_ls = 5
# ------------------------|

spectrum_file = grid.get_flux(emu.grid_points[15])
waves, obs_flux_data = np.loadtxt(
    spectrum_file, usecols=usecols, unpack=True, skiprows=skiprows)
wl_range_data = (wl_range[0] + 10, wl_range[1] - 10)
waves = np.flip(waves)
obs_flux_data = np.flip(obs_flux_data)
# obs_flux_data = gaussian_filter1d(obs_flux_data, 50)
if scale == 'log':
    obs_flux_data = [np.log10(i) for i in obs_flux_data]  # logged scale
    obs_flux_data = np.array(obs_flux_data)
if scale == 'scaled':
    obs_flux_data /= np.mean(obs_flux_data)
indexes = np.where((waves >= wl_range_data[0]) & (waves <= wl_range_data[1]))
waves = waves[indexes[0]]
obs_flux_data = obs_flux_data[indexes[0]]
sigmas = np.zeros(len(waves))
obs_data = Spectrum(waves, obs_flux_data, sigmas=sigmas, masks=None)

model = SpectrumModel(
    f'Grid-Emulator_Files/{emu_file_name}.hdf5',
    data,
    grid_params=list(emu.grid_points[15]),  # [list, of , grid , points]
    Av=0,
    global_cov=dict(log_amp=log_amp, log_ls=log_ls)
)
print(model)

model_flux = model.flux_non_normalised()
plt.plot(waves, model_flux, label='model')
plt.plot(waves, obs_flux_data, label='data')

model2 = SpectrumModel(
    f'Grid-Emulator_Files/{emu_file_name}.hdf5',
    data,
    grid_params=list(emu.grid_points[16]),  # [list, of , grid , points]
    Av=0,
    global_cov=dict(log_amp=log_amp, log_ls=log_ls)
)
model_flux2 = model2.flux_non_normalised()
plt.plot(waves, model_flux2, label="model2")

spectrum_file = grid.get_flux(emu.grid_points[16])
waves, obs_flux_data = np.loadtxt(
    spectrum_file, usecols=usecols, unpack=True, skiprows=skiprows)
wl_range_data = (wl_range[0] + 10, wl_range[1] - 10)
waves = np.flip(waves)
obs_flux_data = np.flip(obs_flux_data)
# obs_flux_data = gaussian_filter1d(obs_flux_data, 50)
if scale == 'log':
    obs_flux_data = [np.log10(i) for i in obs_flux_data]  # logged scale
    obs_flux_data = np.array(obs_flux_data)
if scale == 'scaled':
    obs_flux_data /= np.mean(obs_flux_data)
indexes = np.where((waves >= wl_range_data[0]) & (waves <= wl_range_data[1]))
waves = waves[indexes[0]]
obs_flux_data = obs_flux_data[indexes[0]]
sigmas = np.zeros(len(waves))
obs_data = Spectrum(waves, obs_flux_data, sigmas=sigmas, masks=None)
plt.plot(waves, obs_flux_data, label='data2')
plt.legend()
plt.show()

# %% 7 Pixel Autocorrelation
"""7 Measuring the autocorrelation of pixels """

# ----- Inputs here ------|
# Value of sigma (standard deviation) for the size of filter's gaussian
# kernel in the high pass filter.
high_pass_sigma = 200
# The larger the value, the smoother the flux data but baseline can warp to y = c
# +/- range (aka lags) of the pixels for the autocorrelation plot.
lags = 500
# Specify the standard deviation of the quadratric fit boundaries to
# remove lines.
percent = 0.5
# ------------------------|


def quadratic(x, coefficients):
    """Returns y value for a given x value and coefficients of a quadratic"""
    return coefficients[0] * (x**2) + coefficients[1] * (x) + coefficients[2]


# Alter startfish_flux sigma too (↓) if changing the smoothening term in
# the class KWDGridInterface(GridInterface):
# Smoothened data used for the emulator input
starfish_flux = gaussian_filter1d(raw_flux, 50)
# Really weird bug if not with list!
starfish_flux_original = list(starfish_flux)
# Coefficients of a quadratic fitting routine
coefficients = np.polyfit(waves, starfish_flux, 2)
fit = [quadratic(i, coefficients)
       for i in waves]   # Y values for the quadratic best fit
std1 = np.std(fit)                     # Percent standard deviation of best fit
# Plus 1 standard deviation of best fit for detecting lines
fit_err_plus = fit + (percent * std1)
# Minus 1 standard deviation of best fit for detecting lines
fit_err_minus = fit - (percent * std1)

# Cutting out large emission/absorption lines. Not efficient but too fast to worry about. Reduces large bumps in high pass filter
# Fluxes greater than 1 std have indexes +/- 10 intervals flux values
# replaced by the quadratic fit+noise.
out_of_bounds_indexes = [1 if starfish_flux[i] >= fit_err_plus[i] or starfish_flux[i] <=
                         fit_err_minus[i] else 0 for i in range(len(waves))]  # Detecting lines >1std from fit
for i in range(len(waves)):
    if out_of_bounds_indexes[i] == 1:
        if i < 10:  # if/elif/else statement checking boundaries to stop errors
            waves_limit = range(0, i + 11)
        elif i > (len(waves) - 11):
            waves_limit = range(i - 10, len(waves))
        else:
            waves_limit = range(i - 10, i + 11)
        for j in waves_limit:  # +/- 10 flux intervals
            # change line flux with fit flux plus 5% noise.
            starfish_flux[j] = np.random.normal(fit[j], (0.05 * std1))

# High-pass filter for the starfish flux trend.
smooth_flux = gaussian_filter1d(starfish_flux, high_pass_sigma)
# Starfish data with underlying flux removed.
adj_flux = starfish_flux - smooth_flux

plt.plot(
    waves,
    starfish_flux_original,
    color="grey",
    label='Starfish Flux',
    linewidth=0.5)
plt.plot(
    waves,
    starfish_flux,
    color='red',
    label='Starfish Flux Lines Removed',
    linewidth=0.5)
plt.plot(
    waves,
    fit_err_plus,
    color='orange',
    label=f'Quadratic fit + {percent}$\\sigma$',
    linewidth=0.5)
plt.plot(
    waves,
    fit_err_minus,
    color='orange',
    label=f'Quadratic fit - {percent}$\\sigma$',
    linewidth=0.5)
plt.plot(waves, fit, color='purple', label='Quadratic fit', linewidth=0.5)
plt.plot(waves, smooth_flux, color='green',
         label='Smoothened/High Pass Filter')
if scale == 'linear' or scale == 'scaled':
    plt.plot(
        waves,
        adj_flux,
        color='blue',
        label='Flux Filter Adjusted',
        linewidth=0.5)
plt.xlabel('$\\lambda [\\AA]$')
plt.ylabel('$f_\\lambda$ [$erg/cm^2/s/cm$]')
plt.title('Showing Adjusted Starfish Flux From Being Passed Through \n A High-pass Filter And Quadratic Fit Boundaries To Remove Lines')
plt.legend()
plt.show()
if scale == 'log':
    plt.plot(
        waves,
        adj_flux,
        color='blue',
        label='Flux Filter Adjusted',
        linewidth=0.5)
    plt.xlabel('$\\lambda [\\AA]$')
    plt.ylabel('$f_\\lambda$ [$erg/cm^2/s/cm$]')
    plt.legend()
    plt.show()

# Numpy's autocorrelation method for comparison checks.
mean = np.mean(adj_flux)
var = np.var(adj_flux)
ndata = adj_flux - mean
acorr = np.correlate(ndata, ndata, 'same')
acorr = acorr / var / len(ndata)
pixels = np.arange(len(acorr))
if len(acorr) % 2 == 0:
    pixels = pixels - len(acorr) / 2
else:
    pixels = pixels - len(acorr) / 2 + 0.5
indx = int(np.where(pixels == 0)[0])
lagpixels = pixels[indx - lags:indx + lags + 1]
lagacorr = acorr[indx - lags:indx + lags + 1]
plt.plot(
    lagpixels,
    lagacorr,
    color='green',
    label='Using Numpy Correlate',
    linewidth=0.5)
plt.title(
    f'Adjusted (Starfish Flux - Smoothened {high_pass_sigma}$\\sigma$) Autocorrelation using Full Spectrum')
plt.ylabel('ACF')
plt.xlabel('Pixels')
plt.legend()
plt.show()

# %% 8 Kernel Calculators - to do
"""8 Kernel Calculators"""

# To implement
# %% 8.1 Search grid indexes helper
"""8.1 Search grid point indexes for Stage 9"""
search_grid_points = 1  # Make sure to set to 0 if running a remote session!!!
if search_grid_points == 1:
    print("Search different indexes to find the associated grid point values")
    print("Index range is from 0 to {}".format(len(emu.grid_points) - 1))
    print("Type '-1' to stop searching")
    print("Increasing the index increases the parameters grid points like an odometer")
    print("--------------------------------------------------------------------------")
    print("Names:", emu.param_names)
    print("Description:", [grid.parameters_description(
        model_parameters)[i] for i in emu.param_names])
    while True:
        index = int(input("Enter index input"))
        if index == -1:
            break
        elif 0 <= index <= len(emu.grid_points):
            print("Emulator grid point index {}".format(index))
            print(emu.grid_points[index])
        else:
            print("Not valid input! Type integer between 0 and {} or '-1' to quit".format(
                len(emu.grid_points) - 1))

# %% 9 Assigning model and inital conditions
"""9 Assigning the model and initial model plot"""

# ----- Inputs here ------|
# Natural logarithm of the global covariance's Matern 3/2 kernel amplitude
# log=-52 'linear', log=-8 'log'
log_amp = -7
# 5Natural logarithm of the global covariance's Matern 3/2 kernel lengthscale
log_ls = 5
# ------------------------|

model = SpectrumModel(
    f'Grid-Emulator_Files/{emu_file_name}.hdf5',
    data,
    # [list, of , grid , points]emu.grid_points[119] [-8.95, 10.26, 1.82]
    grid_params=list(emu.grid_points[80]),
    Av=0,
    global_cov=dict(log_amp=log_amp, log_ls=log_ls)
)
print(model)
model.plot(yscale="linear")
model_flux, model_cov = model()
plt.matshow(model._glob_cov, cmap='Greens')
plt.title("Global Covariance Matrix")
plt.colorbar()
plt.show()
plt.matshow(model_cov, cmap='Blues')
plt.title("Sum Covariance Matrix")
plt.colorbar()
plt.show()
model.freeze("Av")
print("-- Model Labels --")
print(model.labels)

# %% 10 Assigning mcmc Priors
# 10
# Default_priors contains a distribution for every possible parameter
# Mostly uniform across grid space bar global_cov being normal
# Change the default distrubtion if you wish something different.
#  WARNING! st.uniform(x, y) is range(x, x+y)
if kgrid == 1:
    default_priors = {
        "param1": st.uniform(1.0e-10, 2.9e-9),
        "param2": st.uniform(4, 28),
        "param3": st.uniform(0.0, 1.0),
        "param4": st.uniform(1.0, 2.0),
        "param5": st.uniform(1e+10, 6e+10),
        "param6": st.uniform(1.0, 5.0),
        "global_cov:log_amp": st.norm(log_amp, 10),
        "global_cov:log_ls": st.uniform(0.1, 10.9),
        "Av": st.uniform(0.0, 1.0)
    }
if shortspec == 1:
    default_priors = {
        # log10 values
        "param1": st.uniform(np.log10(4e-11), (np.log10(3e-9) - np.log10(4e-11))),
        "param2": st.uniform(2, 14),
        "param3": st.uniform(1.0, 2.0),
        "global_cov:log_amp": st.norm(log_amp, 1),
        "global_cov:log_ls": st.uniform(1, 7),
        "Av": st.uniform(0.0, 1.0)
    }

priors = {}  # Selects the priors required from the model parameters used
for label in model.labels:
    priors[label] = default_priors[label]  # if label in default_priors:

# %% 11 Training model with scipy.optimise.minimize(nelder-mead method)
"""11 Training the model"""
initial_simplex = spec.simplex(
    model, priors)  # need to add global covariance hyperparameters
model.train(
    priors,
    options=dict(
        maxiter=1e5,
        disp=True,
        initial_simplex=initial_simplex,
        return_all=True))
print(model)

# %%
model.train(priors, options=dict(maxiter=1e5, disp=True))
print(model)
# %% 12 Saving/plotting trained model
"""12 Saving and plotting the trained model"""
model.plot(yscale="linear")
model.save("Grid-Emulator_Files/Grid_full_MAP.toml")

# %% 12.1 Load trained model
"""12.1 Reloading the trained model"""
model.load("Grid-Emulator_Files/Grid_full_MAP.toml")
model.freeze("global_cov")
print(model.labels)

# %% 13 Set walkers initial positions and mcmc parameters
"""13 Set our walkers and dimensionality"""

# model_ball_initial = {"c1": model["c1"], "c2": model["c2"], "c3":
# model["c3"]} # old code
os.environ["OMP_NUM_THREADS"] = "1"
mp.set_start_method('fork', force=True)

# ----- Inputs here ------|
ncpu = cpu_count() - 2      # Pool CPU's used.
nwalkers = 3 * ncpu         # Number of walkers in the MCMC.
# Maximum iterations of the MCMC if convergence is not reached.
max_n = 400
extra_steps = int(max_n / 10)  # Extra MCMC steps
# ------------------------|

ndim = len(model.labels)
print("{0} CPUs".format(ncpu))
if kgrid == 1:
    default_scales = {"param1": 1e-11, "param2": 1e-2, "param3": 1e-2,
                      "param4": 1e-2, "param5": 1e+9, "param6": 1e-2}
if shortspec == 1:
    default_scales = {"param1": 1e-1, "param2": 1e-1, "param3": 1e-1}

scales = {}  # Selects the priors required from the model parameters used
for label in model.labels:
    scales[label] = default_scales[label]

# Initialize gaussian ball for starting point of walkers
# scales = {"c1": 1e-10, "c2": 1e-2, "c3": 1e-2, "c4": 1e-2}
# model = model_ball_initial
ball = np.random.randn(nwalkers, ndim)
for i, key in enumerate(model.labels):
    ball[:, i] *= scales[key]
    ball[:, i] += model[key]

# %% 14 Running MCMC
"""14 Our objective to maximize and set up our backend/sampler"""

"""def log_prob(P, priors):
    range_min = np.array([1e-10, 4, 0]) #limiting walkers to the grid space
    range_max = np.array([3e-9, 32, 1])
    if np.any(P < range_min) or np.any(P > range_max):
        return -np.inf
    else:
        model.set_param_vector(P)
        return model.log_likelihood(priors)"""


def log_prob(P, priors):
    model.set_param_vector(P)
    return model.log_likelihood(priors)


backend = emcee.backends.HDFBackend(
    "Grid-Emulator_Files/Grid_full_MCMC_chain.hdf5")
backend.reset(nwalkers, ndim)

with Pool(ncpu) as pool:
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_prob, args=(priors,), backend=backend, pool=pool
    )

    index = 0  # Tracking how the average autocorrelation time estimate changes
    autocorr = np.empty(max_n)

    old_tau = np.inf  # This will be useful to testing convergence

    # Now we'll sample for up to max_n steps
    for sample in sampler.sample(ball, iterations=max_n, progress=True):
        # Only check convergence every 10 steps
        if sampler.iteration % 10:
            continue
            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1
        # skip math if it's just going to yell at us
        if np.isnan(tau).any() or (tau == 0).any():
            continue
            # Check convergence
        converged = np.all(tau * 10 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            print(f"Converged at sample {sampler.iteration}")
            break
        old_tau = tau

    sampler.run_mcmc(backend.get_last_sample(), extra_steps, progress=True)

# %% 15 Plot raw MCMC chains
"""15 Showing raw MCMC chain."""
reader = emcee.backends.HDFBackend(
    "Grid-Emulator_Files/Grid_full_MCMC_chain.hdf5")
full_data = az.from_emcee(reader, var_names=model.labels)
flatchain = reader.get_chain(flat=True)
az.plot_trace(full_data)

# %% 16 Discard burn-in
"""16 Discarding MCMC burn-in."""
tau = reader.get_autocorr_time(tol=0)
if m.isnan(tau.max()):
    burnin = 0
    thin = 1
    print(burnin, thin)
else:
    burnin = int(tau.max())
    thin = int(0.3 * np.min(tau))
burn_samples = reader.get_chain(discard=burnin, thin=thin)
log_prob_samples = reader.get_log_prob(discard=burnin, thin=thin)
log_prior_samples = reader.get_blobs(discard=burnin, thin=thin)
dd = dict(zip(model.labels, burn_samples.T))
burn_data = az.from_dict(dd)

# %% 17 Chain trace and summary
"""17 Plotting the mcmc chains without the burn-in section,
   summarise our mcmc run's parameters and analysis,
   plot our posteriors of each paramater,
   produce a corner plot of our parameters."""
az.plot_trace(burn_data)
az.summary(burn_data, round_to=None)
az.plot_posterior(burn_data, [i for i in model.labels])

# %% 18 Cornerplot
"""18 Producing a corner plot of our parameters."""
# See https://corner.readthedocs.io/en/latest/pages/sigmas/
sigmas = ((1 - np.exp(-0.5)), (1 - np.exp(-2)))
corner.corner(
    burn_samples.reshape((-1, len(model.labels))),
    labels=model.labels,
    quantiles=(0.05, 0.16, 0.84, 0.95),
    levels=sigmas,
    show_titles=True,
)

# %% 19 Plot Best Fit MCMC

"""19 We examine our best fit parameters from the mcmc chains,
   plot and save our final best fit model spectrum."""
ee = [np.mean(burn_samples.T[i]) for i in range(len(burn_samples.T))]
ee = dict(zip(model.labels, ee))
model.set_param_dict(ee)
print(model)
model.plot(yscale="linear")
model.save("Grid-Emulator_Files/Grid_full_parameters_sampled.toml")

# %% END
import numpy as np
def generate_random_data():
    """Generate some fake data from the model."""
    np.random.seed(42)
    x = np.linspace(0.1, 30, 1000)
    yerr = 0.1 + 0.5 * np.random.rand(len(x))
    y = model(x) + yerr * np.random.randn(len(x))
    return x, y, yerr

data = generate_random_data()
plt.plot(data[0], data[1], "k.", label="data")
plt.show()
# %%
