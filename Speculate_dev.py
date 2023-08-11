# %% Stage 1) Imports and Speculate Grid Classes/funcitons for 'python'.
# 1) ==========================================================================|

import autopep8
import os
import random
import emcee
import corner
import arviz as az
import math as m
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import scipy.stats as st
import Speculate_addons.Spec_functions as spec

#from alive_progress import alive_it
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

plt.style.use('Solarize_Light2') # Plot Style (⌐▀͡ ̯ʖ▀) (ran twice as buggy-ish)

# %% Stage 2.1) Flux grid space (HDF5) setup and PCA inputs
# 2.1) ========================================================================|

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

n_components = 8              # Alter the number of PCA components used.
# Integer for no. of components or decimal (0.0-1.0) for 0%-100% accuracy.
# -----------------------------------------------------------------------------|

# Sorting parameters by increasing order
model_parameters = sorted(model_parameters)
# Looping through parameters to create a string of numbers for file name
model_parameters_str = ''.join(str(i) for i in model_parameters)

# Selecting the specified grid interface 
if kgrid == 1:
    # Change inclination with usecols[1]
    usecols = (1, 8) # Wavelength, Inclination 8-14 --> 40-70 degrees
    skiprows = 2  # Start of data within file
    grid = KWDGridInterface(
        path='kgrid/sscyg_kgrid090311.210901/',
        usecols=usecols,
        skiprows=skiprows,
        wl_range=wl_range,
        model_parameters=model_parameters)
    inclination = usecols[1] * 5
    emu_file_name = f'Kgrid_emu_{scale}_{usecols[1]*5}inc_{wl_range[0]}-{wl_range[1]}AA_{n_components}comp_{model_parameters_str}'

if shortspec == 1:
    # Change inclination with usecols[1]
    usecols = (1, 21) # Wavelength, Inclination 10-21 --> 30-85 degrees
    skiprows = 81  # Starting point of data within file
    grid = ShortSpecGridInterface(
        path='short_spec_cv_grid/',
        usecols=usecols,
        skiprows=skiprows,
        wl_range=wl_range,
        model_parameters=model_parameters, 
        scale=scale
        )
    inclination = (usecols[1]-4) * 5
    emu_file_name = f'SSpec_emu_{scale}_{(usecols[1]-4) * 5}inc_{wl_range[0]}-{wl_range[1]}AA_{n_components}comp_{model_parameters_str}'
    
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

# %% Stage 2.2) Speculate's spectral data exploration tool (SDET).
# 2.2) ========================================================================|
# The Class should open a new window to allow the user to explore the grid.
%matplotlib qt
grid_viewer = spec.InspectGrid(grid, emu) # Emu (Emulator) optional
# %%
%matplotlib inline

# %% Stage 3) Generating and training a new emulator
# 3) ==========================================================================|

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
# 3.) =========================================================================|

emu.train(method="Nelder-Mead", options=dict(maxiter=1e5, disp=True))
emu.save(f'Grid-Emulator_Files/{emu_file_name}.hdf5')  # Saving the emulator
print(emu)

# %% Stage 4) Plotting the emulator's eigenspectra and weights slice TODO
# 4) ==========================================================================|

#%matplotlib inline
# Inputs: Displayed parameter (1-X), other parameters' fixed index (0-(X-1))
spec.plot_emulator(emu, grid, 1, 0)
# plot_new_eigenspectra(emu, 51)  # <---- Yet to implement

# =============================================================================|

# %% plot_new_eigenspectra function

def plot_new_eigenspectra(emulator, params, filename=None):
    from matplotlib import gridspec
    """
    TODO: Correct the deprecated plotting function from Starfish. 
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
# %% # Stage 5) Plotting the emulator's covariance matrix
# 5) ==========================================================================|

emu = Emulator.load(f"Grid-Emulator_Files/{emu_file_name}.hdf5")
random_grid_point = random.choice(emu.grid_points)
print("Random Grid Point Selection")
print(list(emu.param_names))
print(emu.grid_points[0]) # put emu.grid_points[0] 
print(random_grid_point) # or put: random_grid_point in next line
weights, cov = emu(random_grid_point) # here !!!
X = emu.eigenspectra * (emu.flux_std)
flux = (weights @ X) + emu.flux_mean
emu_cov = X.T @ cov @ X
plt.matshow(emu_cov, cmap='Reds')
plt.title("Emulator Covariance Matrix")
plt.colorbar()
plt.show()
plt.plot(emu.wl, flux)


# %% # Stage 6) Adding observational spectrum as data
# 6) ==========================================================================|

# ---------- Switches here -----------|
# Four methods to select which type of testing spectrum file you want:
# Turn off/on (0/1) to use this method
data_one = 0                # [1] A kgrid grid point
data_two = 0                # [2] A noisy grid point
data_three = 0              # [3] Interpolatation between two grid points
data_four = 1               # [4] A custom test file from python
# -----------------------------------------------------------------------------|

# ----------- Inputs here ------------|
if data_one == 1:
    # File corresponds to grid points in section 2
    # Parameter's point given by the XX number in the name (6 params = 12 digits)
    # 040102000000 = 4.5e-10, 16, 1, 1, 1e10, 3
    file = 'sscyg_k2_040102000001.spec'
    
    waves, fluxes = np.loadtxt(
        f'kgrid/sscyg_kgrid090311.210901/{file}', usecols=usecols, unpack=True, skiprows=skiprows)


if data_two == 1:
    file = 'sscyg_k2_040102000001.spec'  # File naming same as 1)
    noise_std = 0.50                    # Percentage noise (0.05 sigma)
    
    waves, fluxes = np.loadtxt(
        f'kgrid/sscyg_kgrid090311.210901/{file}', usecols=usecols, unpack=True, skiprows=skiprows)
    noise = noise_std * np.std(fluxes)
    for i in range(len(waves)):
        fluxes[i] = np.random.normal(fluxes[i], noise)       


if data_three == 1:
    print('to do') # TODO


# Python v87a formatting. Place python file within kgrid folder/directory
if data_four == 1:
    file = 'run59_WMdot4e-10_d5_vinf2p5.spec' # < CHANGEABLE
    #file = 'runtest_WMdot2e-10_d14_vinf1p5.spec'

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


# %% Stage 7) Measuring the autocorrelation of pixels.
# 7) ==========================================================================|

# ----- Inputs here ------|
# Value of sigma (standard deviation) for the size of filter's gaussian
# kernel in the high pass filter. The larger the value, the smoother the flux
# data but baseline can warp to y = c
high_pass_sigma = 200

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

# %% Stage 8.1) Kernel Calculators"""
# 8.1) ========================================================================|

# TODO: Implement kernel calculators for the emulator's global matrix

# %% Stage 8.2) Search grid indexes helper
# =============================================================================|

spec.search_grid_points(1, emu, grid) # <-- 1/0 switch for on/off

# %% Stage 9) Assigning the model and initial model plot"""
# 9) ==========================================================================|

# ----- Inputs here ------|
# Natural logarithm of the global covariance's Matern 3/2 kernel amplitude
# log=-52 'linear', log=-8 'log'
log_amp = -52
# 5Natural logarithm of the global covariance's Matern 3/2 kernel lengthscale
log_ls = 5
# ------------------------|

model = SpectrumModel(
    f'Grid-Emulator_Files/{emu_file_name}.hdf5',
    data,
    # [list, of , grid , points]emu.grid_points[119] [-8.95, 10.26, 1.82]
    grid_params=list(emu.grid_points[58]),
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

# %% Stage 10) Assigning the mcmc priors
# 10) =========================================================================|

# Default_priors contains a distribution for every possible parameter
# Mostly uniform across grid space bar global_cov being normal
# Change the default distrubtion if you wish something different.
# WARNING! st.uniform(x, y) is range(x, x+y)
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

# %% Stage 11) Training model with scipy.optimise.minimize(nelder-mead method)
# 11) =========================================================================|

# TODO: SIMPLEX - Need to add global covariance hyperparameters
initial_simplex = spec.simplex(model, priors) 
model.train(
    priors,
    options=dict(
        maxiter=1e5,
        disp=True,
        initial_simplex=initial_simplex,
        return_all=True))
print(model)

# %% Stage 11.continued) Continue training the model
# 12) =========================================================================|

model.train(priors, options=dict(maxiter=1e5, disp=True))
print(model)

# %% Stage 12.1) Saving and plotting the trained model
# 12.1) =======================================================================|

model.plot(yscale="linear")
model.save("Grid-Emulator_Files/Grid_full_MAP.toml")

# %% Stage 12.2) Reloading the trained model
# 12.2) =======================================================================|

model.load("Grid-Emulator_Files/Grid_full_MAP.toml")
model.freeze("global_cov")
print(model.labels)

# %% Stage 13) Set walkers initial positions/dimensionality and mcmc parameters
# 13) =========================================================================|
#TODO : Everything onwards need to be improved
#os.environ["OMP_NUM_THREADS"] = "1"
#mp.set_start_method('fork', force=True)

# ----- Inputs here ------|
ncpu = cpu_count() - 2      # Pool CPU's used.
nwalkers = 5 * ncpu         # Number of walkers in the MCMC.
# Maximum iterations of the MCMC if convergence is not reached.
max_n = 4000
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

# %% Stage 14) Running MCMC, maximizing and setting up our backend/sampler
# 14) =========================================================================|

def log_prob(P, priors):
    model.set_param_vector(P)
    return model.log_likelihood(priors)


backend = emcee.backends.HDFBackend(
    "Grid-Emulator_Files/Grid_full_MCMC_chain.hdf5")
backend.reset(nwalkers, ndim)

#with Pool(ncpu) as pool:
sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_prob, args=(priors,), backend=backend
) #pool=pool goes here

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

# %% Stage 15) Plotting raw MCMC chains
# 15) =========================================================================|

reader = emcee.backends.HDFBackend(
    "Grid-Emulator_Files/Grid_full_MCMC_chain.hdf5")
full_data = az.from_emcee(reader, var_names=model.labels)
flatchain = reader.get_chain(flat=True)
walker_plot = az.plot_trace(full_data)

# %% Stage 16) Discarding MCMC burn-in
# 16) =========================================================================|

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

# %% Stage 17) Chain trace and summary
# 17) =========================================================================|

# Plotting the mcmc chains without the burn-in section,
# summarise our mcmc run's parameters and analysis,
# plot our posteriors of each paramater,
# produce a corner plot of our parameters.
burnt_walker_plot = az.plot_trace(burn_data)
print(az.summary(burn_data, round_to=None))
burnt_posteriors = az.plot_posterior(burn_data, [i for i in model.labels])

# %% Stage 18) Cornerplot of our parameters. 
# 18) =========================================================================|

# See https://corner.readthedocs.io/en/latest/pages/sigmas/
sigmas = ((1 - np.exp(-0.5)), (1 - np.exp(-2)))
cornerplot = corner.corner(
    burn_samples.reshape((-1, len(model.labels))),
    labels=model.labels,
    show_titles=True,
    truths=list(emu.grid_points[58])
)
 #   quantiles=(0.05, 0.16, 0.84, 0.95),levels=sigmas,

# %% Stage 19) Plotting Best Fit MCMC parameters
# 19) =========================================================================|

# We examine our best fit parameters from the mcmc chains, plot and save our
# final best fit model spectrum.
ee = [np.mean(burn_samples.T[i]) for i in range(len(burn_samples.T))]
ee = dict(zip(model.labels, ee))
model.set_param_dict(ee)
print(model)
model.plot(yscale="linear")
model.save("Grid-Emulator_Files/Grid_full_parameters_sampled.toml")


# %% Processing-time data sets

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Plotting the training time of the emulator CIV
# Datasets
number_of_pca_components = [2, 3, 4, 5, 6, 7, 8, 9, 10] # x-axis
first_train_time_train = [0.7, 2.8, 11.1, 19.2, 46.1, 118.4, 186.9, 249, 335.3] # y1-axis
second_train_time_train = [0.5, 1.7, 6.4, 19.7, 58.5, 92.4, 154.5, 203.5, 288.3] # y2-axis

number2_of_pca_components = [11, 12, 13, 14, 15] # x-axis
first_train_time_test = [545.5, 966.8, 1178.9, 991.9, 983.3] # y1-axis
second_train_time_test = [490.2, 650.5, 900.0, 825.6, 1063.6] # y2-axis

model_linear = LinearRegression(fit_intercept=True)
model_linear.fit(np.array(number_of_pca_components).reshape(-1, 1), np.array(first_train_time_train).reshape(-1, 1))
time_prediction_linear = model_linear.predict(np.array([11,12,13,14,15]).reshape(-1, 1))
print(model_linear.coef_, 'coefficients')
print(time_prediction_linear, 'prediction')
print(mean_squared_error(first_train_time_test, time_prediction_linear, squared=False), 'MSE')
print(model_linear.score(np.array(number_of_pca_components).reshape(-1, 1), np.array(first_train_time_train).reshape(-1, 1)), 'R^2')

#Linear Regression model to predict the training time for larger PCA components
model = Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression(fit_intercept=True))])
model.fit(np.array(number_of_pca_components).reshape(-1, 1), np.array(first_train_time_train).reshape(-1, 1))
time_prediction = model.predict(np.array([11,12,13,14,15]).reshape(-1, 1))
print(model.named_steps['linear'].coef_)
print(time_prediction)
print(mean_squared_error(first_train_time_test, time_prediction, squared=False), 'MSE')

model2 = Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression(fit_intercept=True))])
model2.fit(np.array(number_of_pca_components).reshape(-1, 1), np.array(second_train_time_train).reshape(-1, 1))
time_prediction2 = model2.predict(np.array([11,12,13,14,15]).reshape(-1, 1))
print(model2.named_steps['linear'].coef_)
print(time_prediction2)
print(mean_squared_error(second_train_time_test, time_prediction2, squared=False), 'MSE')

# Displaying the plots
plt.plot(number_of_pca_components, first_train_time_train, label='First Training', color='red')
plt.plot(number_of_pca_components, second_train_time_train, label='Second Training', color='blue')
plt.plot(number2_of_pca_components, time_prediction_linear, label='Prediction Linear')
plt.plot(number2_of_pca_components, time_prediction, label='Prediction')
plt.plot(number2_of_pca_components, time_prediction2, label='Prediction 2')
plt.plot(number2_of_pca_components, first_train_time_test, label='First Testing', color='red')
plt.plot(number2_of_pca_components, second_train_time_test, label='Second Testing', color='blue')
plt.xlabel('Number of PCA Components')
plt.ylabel('Training Time (s)')
plt.title('Training Time of the Emulator')
plt.legend()
plt.show()
# %%
