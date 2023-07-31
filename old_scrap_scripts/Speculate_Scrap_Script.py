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

# %% Stage 2.1) Inspecting grid
%matplotlib widget
import itertools
from matplotlib.animation import FuncAnimation

unique_combinations = []
for i in itertools.product(*grid.points):
    unique_combinations.append(list(i))

entire_grid_fluxes = []
axis_min_flux = 0
axis_max_flux = 0 # low to ensure first max flux is higher
for parameters in range(len(unique_combinations)):
    flux = grid.load_flux(unique_combinations[parameters])
    entire_grid_fluxes.append(flux)
    max_flux = max(flux) # assigned for 1 function evaluation, not 3
    if max_flux > axis_max_flux:
        axis_max_flux = max_flux # finding the highest flux value
    
fig, ax = plt.subplots()
#animation of the changing unique combinations indexes' spectrum
def frame_function(i):
    ax.clear()
    ax.plot(grid.wl, entire_grid_fluxes[i], legend=)
    ax.set_title(f"Grid Point {i}'s Spectrum")
    ax.set_xlabel("Wavelength ($\AA$)")
    ax.set_ylabel("Flux")
    ax.set_xlim(min(grid.wl), max(grid.wl))
    ax.set_ylim(axis_min_flux, axis_max_flux)
    return ax
    
animation = FuncAnimation(fig, frame_function, frames=len(unique_combinations), interval=100)
plt.show()
# %%

# %% Stage 2.1) Inspecting grid space semi working
%matplotlib qt
import itertools
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

class InspectGrid:
    def __init__(self, grid):
        self.grid = grid # adding grid to class
        
        # Creating a list of all unique combinations of parameters
        self.unique_combinations = []
        for i in itertools.product(*grid.points):
            self.unique_combinations.append(list(i))
        
        # Finding the min/max flux values for the entire grid space    
        self.entire_grid_fluxes = []
        self.axis_min_flux = 0
        self.axis_max_flux = 0 # low to ensure first max flux is higher
        for parameters in range(len(self.unique_combinations)):
            flux = grid.load_flux(self.unique_combinations[parameters])
            self.entire_grid_fluxes.append(flux)
            max_flux = max(flux) # assigned for 1 function evaluation, not 3
            if max_flux > self.axis_max_flux:
                self.axis_max_flux = max_flux # finding the highest flux value

        self.fig, self.ax = plt.subplots() # initialising plot
    
        
        # Adding grid point slider
        self.fig.subplots_adjust(bottom=0.25) # adjusting plot size to fit slider
        grid_axis = self.fig.add_axes([0.25, 0.1, 0.55, 0.03]) # Slider shape
        self.grid_slider = Slider(grid_axis, 'Grid Point', 0, len(self.unique_combinations), valinit=0, valstep=1)
        
        self.grid_slider.on_changed(self.update_plot)
        
        # Pause animation on mouse click
        self.fig.canvas.mpl_connect('button_press_event', self.toggle_pause)
        self.animation = FuncAnimation(
            self.fig,
            self.plot_function,
            frames=len(self.unique_combinations),
            interval=100,
            )
        self.animation.running = True
        plt.show()
    
    def plot_function(self, i):
        """Function to be called for each frame of the animation"""
        self.frame = i
        self.ax.clear() # clearing the previous frame
        self.ax.set_title(f"Grid Point {self.frame}'s Spectrum")
        self.ax.set_xlabel("Wavelength ($\AA$)")
        self.ax.set_ylabel("Flux")
        self.ax.set_xlim(min(grid.wl), max(grid.wl))
        self.ax.set_ylim(self.axis_min_flux, self.axis_max_flux*1.1)
        self.ax.plot(grid.wl, self.entire_grid_fluxes[self.frame], label=self.unique_combinations[self.frame])
        self.ax.legend()
        self.grid_slider.set_val(self.frame)
    
    def toggle_pause(self, event, *args, **kwargs):
        """Pauses and unpauses the animation on mouse click"""
        (xm,ym),(xM,yM) = self.grid_slider.label.clipbox.get_points()
        if xm < event.x < xM and ym < event.y < yM:
            self.animation.pause()
            self.animation.running = False
        elif self.animation.running:
            self.animation.pause()
            self.animation.running = False
        else:
            self.animation.resume()
            self.animation.running = True
            
    def update_plot(self, extra):
        """A slider to change the grid point frame number"""
        num = self.grid_slider.val
        self.ax.clear() # clearing the previous frame
        self.ax.set_title(f"Grid Point {num}'s Spectrum")
        self.ax.set_xlabel("Wavelength ($\AA$)")
        self.ax.set_ylabel("Flux")
        self.ax.set_xlim(min(grid.wl), max(grid.wl))
        self.ax.set_ylim(self.axis_min_flux, self.axis_max_flux*1.1)
        self.ax.plot(grid.wl, self.entire_grid_fluxes[num], label=self.unique_combinations[num])
        self.ax.legend()
        self.fig.canvas.draw_idle()
        self.animation
        """
        self.animation.pause()
        self.animation.running = False
        self.ax = self.plot_function(self.grid_slider.val)
        self.fig.canvas.draw_idle()
        """

grid_viewer = InspectGrid(grid)
# %% Backup of Stage 2.1) Inspecting grid space
%matplotlib qt
import itertools
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

class InspectGrid:
    def __init__(self, grid):
        self.grid = grid # adding grid to class
        
        # Creating a list of all unique combinations of parameters
        self.unique_combinations = []
        for i in itertools.product(*grid.points):
            self.unique_combinations.append(list(i))
        
        # Finding the min/max flux values for the entire grid space    
        self.entire_grid_fluxes = []
        self.axis_min_flux = 0
        self.axis_max_flux = 0 # low to ensure first max flux is higher
        for parameters in range(len(self.unique_combinations)):
            flux = grid.load_flux(self.unique_combinations[parameters])
            self.entire_grid_fluxes.append(flux)
            max_flux = max(flux) # assigned for 1 function evaluation, not 3
            if max_flux > self.axis_max_flux:
                self.axis_max_flux = max_flux # finding the highest flux value

        self.fig, self.ax = plt.subplots() # initialising plot
    
        
        # Adding grid point slider
        self.fig.subplots_adjust(bottom=0.25) # adjusting plot size to fit slider
        grid_axis = self.fig.add_axes([0.25, 0.1, 0.55, 0.03]) # Slider shape
        self.grid_slider = Slider(grid_axis, 'Grid Point', 0, len(self.unique_combinations), valinit=0, valstep=1)
        
        self.grid_slider.on_changed(self.update_plot)
        
        # Pause animation on mouse click
        self.fig.canvas.mpl_connect('button_press_event', self.toggle_pause)
        self.animation = FuncAnimation(
            self.fig,
            self.plot_function,
            frames=len(self.unique_combinations),
            interval=100,
            )
        self.animation.running = True
        plt.show()
    
    def plot_function(self, i):
        """Function to be called for each frame of the animation"""
        self.frame = i
        self.ax.clear() # clearing the previous frame
        self.ax.set_title(f"Grid Point {self.frame}'s Spectrum")
        self.ax.set_xlabel("Wavelength ($\AA$)")
        self.ax.set_ylabel("Flux")
        self.ax.set_xlim(min(grid.wl), max(grid.wl))
        self.ax.set_ylim(self.axis_min_flux, self.axis_max_flux*1.1)
        self.ax.plot(grid.wl, self.entire_grid_fluxes[self.frame], label=self.unique_combinations[self.frame])
        self.ax.legend()
        self.grid_slider.set_val(self.frame)
    
    def toggle_pause(self, event, *args, **kwargs):
        """Pauses and unpauses the animation on mouse click"""
        (xm,ym),(xM,yM) = self.grid_slider.label.clipbox.get_points()
        if xm < event.x < xM and ym < event.y < yM:
            self.animation.pause()
            self.animation.running = False
        elif self.animation.running:
            self.animation.pause()
            self.animation.running = False
        else:
            self.animation.resume()
            self.animation.running = True
            
    def update_plot(self, extra):
        """A slider to change the grid point frame number"""
        num = self.grid_slider.val
        self.ax.clear() # clearing the previous frame
        self.ax.set_title(f"Grid Point {num}'s Spectrum")
        self.ax.set_xlabel("Wavelength ($\AA$)")
        self.ax.set_ylabel("Flux")
        self.ax.set_xlim(min(grid.wl), max(grid.wl))
        self.ax.set_ylim(self.axis_min_flux, self.axis_max_flux*1.1)
        self.ax.plot(grid.wl, self.entire_grid_fluxes[num], label=self.unique_combinations[num])
        self.ax.legend()
        self.fig.canvas.draw_idle()
        #self.animation.
        """
        self.animation.pause()
        self.animation.running = False
        self.ax = self.plot_function(self.grid_slider.val)
        self.fig.canvas.draw_idle()
        """

grid_viewer = InspectGrid(grid)


#------------------------------------------------------------------------------|
#------------------------------------------------------------------------------|
#%% Stage 5 Debugging code for Speculate 24/07/2023 for plotting spectra and
# correlation matrixes. 

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

#------------------------------------------------------------------------------|
#------------------------------------------------------------------------------|

# %% DEBUG: Data spectrum data file and emulation model overplotting (normalisation)
# 24/07/2023 - Stage 6 
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

# %% 24/07/2023 - Stage 13: old code
# model_ball_initial = {"c1": model["c1"], "c2": model["c2"], "c3":
# model["c3"]} # old code

#Stage 14: old code - limiting parameter space forcefully. 
"""def log_prob(P, priors):
    range_min = np.array([1e-10, 4, 0]) #limiting walkers to the grid space
    range_max = np.array([3e-9, 32, 1])
    if np.any(P < range_min) or np.any(P > range_max):
        return -np.inf
    else:
        model.set_param_vector(P)
        return model.log_likelihood(priors)"""
        
# %% 
    
# %%
