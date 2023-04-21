# %% 
"""1 Austen's Starfish Version for 'Python' in python, built upon from Surya and Czekala et al. 2015"""

# Importing modules and custom functions

import numpy as np
import Starfish
from Starfish.grid_tools import PHOENIXGridInterfaceNoAlpha
from Starfish.grid_tools import download_PHOENIX_models
from Starfish.grid_tools.instruments import SPEX
from Starfish.grid_tools import HDF5Creator
from Starfish.emulator import Emulator
from Starfish.emulator.plotting import plot_emulator
from Starfish.emulator.plotting import plot_eigenspectra
import matplotlib.pyplot as plt
from Starfish.spectrum import Spectrum
from Starfish.models import SpectrumModel
import scipy.stats as st
import emcee
import arviz as az
import corner
import time
import matplotlib
import itertools
from tqdm import tqdm
import random
import math as m
#Surya
import h5py
import os
from Starfish.grid_tools import GridInterface
from scipy.ndimage import gaussian_filter1d
from multiprocessing import Pool
from multiprocessing import cpu_count

class KWDGridInterface(GridInterface):
    """"
    An Interface to the KWD grid produced by PYTHON simulation.
    
    The wavelengths in the spectra are in Angstrom and fluxes in erg/s/cm^2/cm
    
    Parameters of model
    -------------------
    1) wind.mdot (msol/yr)
    2) kn.d
    3) kn.mdot_r_exponent
    4) kn.v_infinity (in_units_of_vescape)
    5) kn.acceleration_length (cm)
    6) kn.acceleration_exponent
    
    Optional parameters
    -------------------
    angle of inclination (taken to be 40.0 for this analysis)
    
    """
    
    def __init__(self, path, air=False, wl_range=(1200,1800), model_parameters=(1,2,3)):
        """"
        Initialises an empty grid with parameters and wavelengths.
        
        Parameters
        ----------
        path : str or path-like
            The path of the base of the KWD library.
        air : bool, optional
            Whether the wavelengths are measured in air or not. Default is False
            (Required due to implementation of inherited GridInterface class)
        wl_range : tuple, optional
            The (min, max) of the wavelengths in AA. Default is (1200, 1400) for testing.
        model_parameters : tuple, optional
            Specifiy the parameters you wish to fit by adding intergers to the tuple. 
        """
        # The grid points in the parameter space are defined, 
        # param_1-6_points correspond to the model parameters defined at the top in the respective order.
        self.model_parameters = model_parameters
        points = []
        if 1 in model_parameters:
            param_points_1 = np.array([1e-10, 1.5e-10, 2.1e-10, 3.1e-10, 4.5e-10, 6.6e-10, 9.7e-10, 1.4e-09, 2.1e-09, 3e-09])
            points.append(param_points_1)
        if 2 in model_parameters:
            param_points_2 = np.array([4, 16, 32])
            points.append(param_points_2)
        if 3 in model_parameters:
            param_points_3 = np.array([0, 0.5, 1])
            points.append(param_points_3)
        if 4 in model_parameters:
            param_points_4 = np.array([1, 2, 3])
            points.append(param_points_4)
        if 5 in model_parameters:
            param_points_5 = np.array([1e+10, 3e+10, 7e+10])
            points.append(param_points_5)
        if 6 in model_parameters:
            param_points_6 = np.array([1, 3, 6])
            points.append(param_points_6)
            
        param_names = ["param{}".format(number) for number in model_parameters] # formatting the parameter names

        # Inititalising the GridInterface with the KWD parameters.
        super().__init__(
            name='KWD',
            param_names=param_names,
            points=points,
            wave_units='AA',
            flux_units='erg/s/cm^2/cm',
            air=air,
            wl_range=wl_range,
            path=path,
        )
        
        # The wavelengths for which the fluxes are measured are retrieved.
        try:
            wls_fname = os.path.join(self.path, 'sscyg_k2_000000000000.spec')
            wls = np.loadtxt(wls_fname, delimiter=' ', usecols=(1), skiprows=2)
            wls = np.flip(wls)
        except:
            raise ValueError("Wavelength file improperly specified")
        
        # Truncating to the wavelength range to the provided values.
        self.wl_full = np.array(wls, dtype=np.float64) #wls[::-1]
        self.ind = (self.wl_full >= self.wl_range[0]) & (
            self.wl_full <= self.wl_range[1])
        self.wl = self.wl_full[self.ind]
        
        #self.rname = ""
        #self.full_rname = os.path.join(self.path, self.rname)
        
        
    def get_flux(self, params):
        """"
        Constructs path of datafile corresponding to parameters passed.
        
        Parameters
        ----------
        params : ndarray
            Contains the parameters of a required grid point.
            
        Returns
        -------
        str
            The path of the datafile corresponding to the input parameters.
            
        """
        #param_names = ["c{}".format(number) for number in parameter_numbers]
        # dict file name format for different param values.
        param1_name = {1e-10: '00', 1.5e-10: '01', 2.1e-10: '02', 3.1e-10: '03', 4.5e-10: '04',
                       6.6e-10: '05', 9.7e-10: '06', 1.4e-09: '07', 2.1e-09: '08', 3e-09: '09'}
        param2_name = {4: '00', 16: '01', 32: '02'}
        param3_name = {0: '00', 0.5: '01', 1: '02'}
        param4_name = {1: '00', 2: '01', 3: '02'}
        param5_name = {1e+10: '00', 3e+10: '01', 7e+10: '02'}
        param6_name = {1: '00', 3: '01', 6: '02'}
        all_names = [param1_name, param2_name, param3_name, param4_name, param5_name, param6_name] # Can be improved with dictionary
        param_numbers = params
        base = self.path + 'sscyg_k2_'
        for loop in range(len(all_names)):
            if "param{}".format(loop+1) in self.param_names:
                base += all_names[loop][param_numbers[loop]]
            else:
                base += '00'
                param_numbers = np.insert(param_numbers, loop, 0)          
        return base + '.spec'

    def parameters_description(self, model_parameters):
        """Provides a description of the model parameters used.

        Args:
            model_parameters (tuple): Numbers of the parameters used in the model.

        Returns:
            dictionary: Description of the 'paramX' name
        """
        dictionary = {
            1:"wind.mdot (msol/yr)",
            2:"kn.d",
            3:"kn.mdot_r_exponent",
            4:"kn.v_infinity (in_units_of_vescape)",
            5:"kn.acceleration_length (cm)",
            6:"n.acceleration_exponent"
            } #Description of the paramters
        parameters_used = {}
        for i in model_parameters:
            parameters_used["param{}".format(i)] = dictionary[i]
        return parameters_used
        
    def load_flux(self, parameters, header=False, norm=False, angle_inc=0):
        """"
        Returns the Flux of a given set of parameters.
        
        Parameters
        ----------
        parameters : ndarray
            Contains parameters of a required grid point
            
        header : bool
            Whether to attach param values on return
            
        norm : bool
            Whether to normalise the return flux (left unimplemented)
            
        angle_inc : int
            Angle of inclination, takes values between 0 to 6 corresponding
            to values between 40.0 and 70.0 with 5.0 degree increment.
            
        Returns
        -------
        ndarray
            List of fluxes in the wavelength range specified on initialisation
            
        dict (Optional)
            Dictionary of parameter names and values
        
        """
        from scipy.ndimage import gaussian_filter1d # Instead of normalising, a 1d gaussian smoothing filter is applied 
        flux = np.loadtxt(self.get_flux(parameters), usecols=(8+angle_inc), skiprows=2)
        flux = np.flip(flux)
        flux = gaussian_filter1d(flux, 50)
        flux = np.log10(flux) #log
        
        hdr = {'c0' : angle_inc} # Header constructed (channel 0 corresponds to angle of inclination)
        for i in range(len(self.param_names)):
            hdr[self.param_names[i]] = parameters[i]

        if(header):
            return flux[self.ind], hdr
        else:
            return flux[self.ind]

def plot_emulator(emulator, model_parameters, not_fixed):
    
    # Placing the grid points values within a dictionary, keyed as 'params{}'
    variables = {}
    for loop in model_parameters:
        variables["param{}".format(loop)] = np.unique(emulator.grid_points[:, model_parameters.index(loop)])
        
    # Creating a custom itertools.product routine which can dynamically input the free varying parameter
    # and the length of the number of parameters depending on what is specified. 
    # params = np.array(list(itertools.product(T, logg[:1], Z[:1]))) # <-- starfish original
    not_fixed_index = model_parameters.index(not_fixed) # Converting parameter number to index position
    params = []
    temp = [variables[emulator.param_names[j]] for j in range(len(variables))] # Creating list from dictionary
    # New list fixing the other parameters on the first grid point↓
    temp2 = [np.array(temp[i]) if i==not_fixed_index else temp[i][0] for i in range(len(temp))] # <--- Change the 0 in temp[i][0] if you wish to see other fixed parameters indexes
    for j in range(len(temp2[not_fixed_index])): # Itertools.product calculation into the same original formatting
        params.append(tuple([temp2[i][j] if temp2[i].size>1 else temp2[i] for i in range(len(temp2))]))
    params = np.array(params)
    
    idxs = np.array([emulator.get_index(p) for p in params])
    weights = emulator.weights[idxs.astype("int")].T
    if emulator.ncomps < 4:
        fix, axes = plt.subplots(emulator.ncomps, 1, sharex=True, figsize=(8,(emulator.ncomps-1)*2))
    else:
        fix, axes = plt.subplots(
            int(np.ceil(emulator.ncomps/2)), 2, sharex=True, figsize=(13,(emulator.ncomps-1)*2),)
    axes = np.ravel(np.array(axes).T)
    [ax.set_ylabel(f"$weights_{i}$") for i, ax in enumerate(axes)]
    
    param_x_axis = np.unique(emulator.grid_points[:,not_fixed_index])
    for i, w in enumerate(weights):
        axes[i].plot(param_x_axis, w, "o")
        
    # Again as above, dynamical input for the gaussian process errors to be plotted for the specified parameter
    param_x_axis_test = np.linspace(param_x_axis.min(), param_x_axis.max(), 100)
    temp2[not_fixed_index] = param_x_axis_test
    Xtest = []
    for j in range(len(temp2[not_fixed_index])):
        Xtest.append(tuple([temp2[i][j] if temp2[i].size>1 else temp2[i] for i in range(len(temp2))]))
    Xtest = np.array(Xtest)
    mus = []
    covs = []
    for X in Xtest:
        m, c = emulator(X)
        mus.append(m)
        covs.append(c)
    mus = np.array(mus)
    covs = np.array(covs)
    sigs = np.sqrt(np.diagonal(covs, axis1=-2, axis2=-1))
    for i, (m, s) in enumerate(zip(mus.T, sigs.T)):
        axes[i].plot(param_x_axis_test, m, "C1")
        axes[i].fill_between(param_x_axis_test, m - 2 * s, m + 2 * s, color="C1", alpha = 0.4)
        axes[i].set_xlabel("Parameter{}".format(not_fixed))
    plt.suptitle("Weights for Parameter{} with the other parameters fixed to their zero index grid point".format(not_fixed), fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

def newlagtimes(x,y,maxlags):
    """Normalised Correlation of two lightcurve arrays using the entire lightcurve dataset
    Austen's previously made cross correlation routine. Auto formats datasets. Instant output. 'Same' method.
    Use the same dataset for x and y if you wish to find the Autocorrelation
    
    Args:
        x (numpy array): 1D array of timeseries/data for cross-correlation
        y (numpy array): 1D array of timeseries/data for cross-correlation. 
        Input the same x data here for auto-correlation
        maxlags (int): Maximum number of lags you wish to be generated. 
        The maximum this can be is the length of your array
    """
    
    def ccf(x,y):
        """CCF for two equally sized arrays"""
        N=0
        summation=0
        meanx = np.average(x)
        meany = np.average(y)
        variancex = np.var(x)
        variancey = np.var(y)
        for valuex, valuey in zip(x, y):
            N = N+1
            summation = summation + ((valuex - meanx)*(valuey - meany))
        ratio = summation/(np.sqrt(variancex*variancey)*N)
        return ratio   
    
    resultarray = [] #initalising parameters
    lagtimearray = np.arange(-maxlags, maxlags+1, 1) #lag times for plot
    lenlagarray = np.arange(0, len(lagtimearray),1) #iteration loop
    if len(x) < len(y):                             #x array is smaller than y array and moves across y
        for iteration in lenlagarray:
            if iteration > maxlags:                 #positive lags
                index = int(iteration - maxlags)    #iterating index for the array slicing 
                ysliced = y[index:]                 #cutting up the arrays for equal lenghts
                if len(ysliced) < len(x):
                    xsliced = x[:len(ysliced)]
                    resultarray = np.append(resultarray, ccf(xsliced, ysliced)) #performing ccf
                else:
                    ysliced = y[index:len(ysliced)]
                    resultarray = np.append(resultarray, ccf(x, ysliced)) #permforming ccf
                    
            else: #negative lags
                index = int(-1*(iteration-maxlags))     #iterating index for the array slicing
                xsliced = x[index:]                     #cutting front of the array off for negative lags
                ysliced = y[:len(xsliced)]              #cutting the end off of other for equal lenght
                resultarray = np.append(resultarray, ccf(xsliced, ysliced)) #permoforming ccf
                
            
    elif len(x) > len(y): #x array is larger than y array and moves across y
        for iteration in lenlagarray:
            if iteration > maxlags:                 #positive lags
                index = int(iteration - maxlags)    #iterating index for the array slicing 
                ysliced = y[index:]                 #cutting up the arrays for equal lenghts
                xsliced = x[:len(ysliced)]
                resultarray = np.append(resultarray, ccf(xsliced, ysliced)) #performing ccf
                    
            else: #negative lags
                index = int(-1*(iteration-maxlags))     #iterating index for the array slicing
                xsliced = x[index:]                     #cutting front of the array off for negative lags
                if len(xsliced) < len(y):    
                    ysliced = y[:len(xsliced)]              #cutting the end off of other for equal lenght
                    resultarray = np.append(resultarray, ccf(xsliced, ysliced)) #permoforming ccf
                else:
                    xsliced = x[index:len(xsliced)]
                    resultarray = np.append(resultarray, ccf(xsliced, y)) #permoforming ccf
                    
    else:
        for iteration in lenlagarray:
            if iteration > maxlags:                 #positive lags
                index = int(iteration - maxlags)    #iterating index for the array slicing 
                ysliced = y[index:]                 #cutting up the arrays for equal lenghts
                xsliced = x[:len(ysliced)]
                resultarray = np.append(resultarray, ccf(xsliced, ysliced)) #performing ccf            
            
            else: #negative lags
                index = int(-1*(iteration-maxlags))     #iterating index for the array slicing
                xsliced = x[index:]                     #cutting front of the array off for negative lags
                ysliced = y[:len(xsliced)]              #cutting the end off of other for equal lenght
                resultarray = np.append(resultarray, ccf(xsliced, ysliced)) #permoforming ccf  
    return lagtimearray, resultarray

"""<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>"""
"""<><><><><><><><><><><><><><><><><><><><><><>START OF CODE SCRIPT<><><><><><><><><><><><><><><><><><><><><><>"""
"""<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>"""

# %%
"""2) - Set up your grid space! - 
        Parameters of model
        -------------------
        1) wind.mdot (msol/yr)
        2) kn.d
        3) kn.mdot_r_exponent
        4) kn.v_infinity (in_units_of_vescape)
        5) kn.acceleration_length (cm)
        6) kn.acceleration_exponent
"""
    
### ----- Inputs here ------|
model_parameters=(1,2,5)  # Parameters shown above
wl_range = (900,1800)       # Wavelength range of your emulator grid space kgrid:(min, max)=(876,1824)
### ------------------------|

model_parameters = sorted(model_parameters) # Sorting parameters by increasing order
grid = KWDGridInterface(path='kgrid/sscyg_kgrid090311.210901/', wl_range=wl_range, model_parameters=model_parameters) # Grid space function

# Faster processing with python's .spec files into a hdf5 file
keyname = ["param{}{{}}".format(i) for i in model_parameters]   # Auto-generated keyname's from parameters
keyname = ''.join(keyname)                                      # Keyname required to integrate with Starfish 
creator = HDF5Creator(grid, 'Grid-Emulator_Files/Grid_full.hdf5', key_name=keyname, wl_range=wl_range) # Processing grid function
creator.process_grid()

# %% 
"""3) Generating and training the emulator grid for modelling"""

### ----- Inputs here ------|
n_components = 6           # Alter the number of components in the PCA decomposition  
                            # Integer for no. of components or decimal (0.0-1.0) for 0%-100% accuracy. 
### ------------------------|

emu = Emulator.from_grid('Grid-Emulator_Files/Grid_full.hdf5', n_components=n_components, svd_solver="full") # Emulator grid function
emu.train(options=dict(maxiter=1e5)) # Training the emulator grid, maximum iterations for the scipy.optimise.minimise routine
emu # Displays the trained emulator's parameters

# %%
"""4) Plotting and saving the emulator"""
#plot_eigenspectra(emu, 0) # <---- Yet to implement
plot_emulator(emu, model_parameters, 1) # <---- Change which parameter is the x-axis, values in section 2
emu.save('Grid-Emulator_Files/Emulator_full.hdf5')

# %%
"""5) If you want to see the emulator covariance matrix for a grid point, run this cell"""

emu = Emulator.load("Grid-Emulator_Files/Emulator_full.hdf5")
random_grid_point = random.choice(emu.grid_points)
print("Random Grid Point Selection")
print(list(emu.param_names))
print(random_grid_point)
weights, cov = emu(random_grid_point)
X = emu.eigenspectra * (emu.flux_std)
flux = (weights @ X) + emu.flux_mean
emu_cov = X.T @ cov @ X
plt.matshow(emu_cov, cmap='Reds')
plt.title("Emulator Covariance Matrix")
plt.colorbar()
plt.show()

# %% 
"""5.5) Emulator's correlation matrix with grid points data, debugging training convergence issues"""

correlation_matrix = np.empty((len(emu.grid_points), len(emu.grid_points)))
for i in tqdm(range(len(emu.grid_points))): # Iterating across the spectral file grid points(changing file names)
    spectrum_file = grid.get_flux(emu.grid_points[i])
    waves, obs_flux_data = np.loadtxt(spectrum_file, usecols=(1, 8), unpack = True)
    wl_range_data = (wl_range[0]+50, wl_range[1]-50)
    waves = np.flip(waves)
    obs_flux_data = np.flip(obs_flux_data)
    obs_flux_data = gaussian_filter1d(obs_flux_data, 50)
    obs_flux_data = [np.log10(i) for i in obs_flux_data] #log
    obs_flux_data = np.array(obs_flux_data) #log
    indexes = np.where((waves >= wl_range_data[0]) & (waves <= wl_range_data[1]))
    waves = waves[indexes[0]]
    obs_flux_data = obs_flux_data[indexes[0]]
    sigmas = np.zeros(len(waves))
    obs_data = Spectrum(waves, obs_flux_data, sigmas=sigmas, masks=None)
    for j in range(len(emu.grid_points)): #Iterating across the emulated files
        model = SpectrumModel(
            'Grid-Emulator_Files/Emulator_full.hdf5',
            obs_data,
            grid_params=list(emu.grid_points[j]), #[list, of , grid , points]
            Av=0,
            global_cov=dict(log_amp=-8, log_ls=5)
            )
        model_flux = model.flux_non_normalised()
        a = np.corrcoef(obs_flux_data, model_flux)
        correlation_matrix[i][j] = a[0][1]

   
# %%
"""5.75 Plotting the correlation coefficient matrix"""        
plt.matshow(correlation_matrix)
plt.title("Correlation coefficient matrix of the emulated spectral grid point \n compared to the real spectral grid point")
plt.colorbar()
plt.xlabel('Emulator Grid Points')
plt.ylabel('Data Grid Points')

# %% 
"""6) Producing test spectrum data for the emulator comparisons."""

"""Four methods to select which type of test spectrum file you want:
    1) A grid point 
    2) A noisy version of a grid point
    3) Interpolation between a few grid point spectrum (method detailed in stage 6)
    4) A custom test file from python
"""
### ----- Switches here ----|
                            # Turn off/on [0,1] to use this method
data_one = 1                # 1)                        
data_two = 0                # 2)                                               
data_three = 0              # 3) 
data_four = 0               # 4) 
### ------------------------|

### ----------- Inputs here ------------|
# 1)
if data_one == 1:
    file = 'sscyg_k2_040102000001.spec' # < File corresponds to grid points in section 2 
                                        # < Parameter's point given by the XX number in the name (6 params = 12 digits)
                                        # < 040102000000 = 4.5e-10, 16, 1, 1, 1e10, 3
# 2)                          
if data_two == 1:
    file = 'sscyg_k2_040102000001.spec' # < File naming same as 1)
    noise_std = 0.50                    # < Percentage noise (0.05 sigma)
# 3)                          
if data_three == 1:
    print('to do')
# 4)
if data_four == 1:
    file = 'ss_cyg_1.spec'              # Python v87a formatting. Place python file within kgrid folder/directory

### ------------------------------------|

if data_one == 1: 
    waves, fluxes = np.loadtxt(f'kgrid/sscyg_kgrid090311.210901/{file}', usecols=(1,8), unpack = True)
    
if data_two == 1: 
    waves, fluxes = np.loadtxt(f'kgrid/sscyg_kgrid090311.210901/{file}', usecols=(1,8), unpack = True)
    noise = noise_std * np.std(fluxes)
    for i in range(len(waves)):
        fluxes[i] = np.random.normal(fluxes[i], noise)
        
if data_three == 1:
    print('to do')
    
if data_four == 1:
    waves, fluxes = np.loadtxt(f'kgrid/{file}', unpack = True, usecols=(1,10), skiprows=74)

# Data manipulation/truncation into correct format.
wl_range_data = (wl_range[0]+50, wl_range[1]-50)
waves = np.flip(waves)
fluxes = np.flip(fluxes)
fluxes = gaussian_filter1d(fluxes, 50)
fluxes = [np.log10(i) for i in fluxes] #log
fluxes = np.array(fluxes) #log
indexes = np.where((waves >= wl_range_data[0]) & (waves <= wl_range_data[1]))
waves = waves[indexes[0]]
fluxes = fluxes[indexes[0]]
raw_flux = list(fluxes)
sigmas = np.zeros(len(waves))
data = Spectrum(waves, fluxes, sigmas=sigmas, masks=None)
data.plot(yscale="linear")


# %%
"""7 Measuring the autocorrelation of pixels """

### ----- Inputs here ------|
high_pass_sigma = 200       # Value of sigma (standard deviation) for the size of filter's gaussian kernel in the high pass filter. 
                            # The larger the value, the smoother the flux data but baseline can warp to y = c 
lags = 500                  # +/- range (aka lags) of the pixels for the autocorrelation plot.
percent = 0.5               # Specify the standard deviation of the quadratric fit boundaries to remove lines. 
### ------------------------|

def quadratic(x, coefficients):
    """Returns y value for a given x value and coefficients of a quadratic"""
    return coefficients[0]*(x**2) + coefficients[1]*(x) + coefficients[2]


# Alter startfish_flux sigma too (↓) if changing the smoothening term in the class KWDGridInterface(GridInterface):
starfish_flux = gaussian_filter1d(raw_flux, 50)      # Smoothened data used for the emulator input
starfish_flux_original = list(starfish_flux)        # Really weird bug if not with list! 
coefficients = np.polyfit(waves, starfish_flux, 2)  # Coefficients of a quadratic fitting routine
fit = [quadratic(i, coefficients) for i in waves]   # Y values for the quadratic best fit
std1 = np.std(fit)                     # Percent standard deviation of best fit
fit_err_plus = fit + (percent * std1)  # Plus 1 standard deviation of best fit for detecting lines
fit_err_minus = fit - (percent * std1) # Minus 1 standard deviation of best fit for detecting lines

# Cutting out large emission/absorption lines. Not efficient but too fast to worry about. Reduces large bumps in high pass filter
# Fluxes greater than 1 std have indexes +/- 10 intervals flux values replaced by the quadratic fit+noise. 
out_of_bounds_indexes = [1 if starfish_flux[i]>=fit_err_plus[i] or 
                         starfish_flux[i]<=fit_err_minus[i] else 0 for i in range(len(waves))] # Detecting lines >1std from fit
for i in range(len(waves)):
    if out_of_bounds_indexes[i] == 1:
        if i<10: # if/elif/else statement checking boundaries to stop errors
            waves_limit = range(0, i+11)
        elif i>(len(waves)-11):
            waves_limit = range(i-10, len(waves))
        else:
            waves_limit = range(i-10, i+11)
        for j in waves_limit: # +/- 10 flux intervals
                starfish_flux[j] = np.random.normal(fit[j], (0.05*std1)) # change line flux with fit flux plus 5% noise.    
                
smooth_flux = gaussian_filter1d(starfish_flux, high_pass_sigma) # High-pass filter for the starfish flux trend. 
adj_flux = starfish_flux - smooth_flux # Starfish data with underlying flux removed. 

plt.plot(waves, starfish_flux_original, color = "grey", label='Starfish Flux', linewidth=0.5)
plt.plot(waves, starfish_flux, color='red', label='Starfish Flux Lines Removed', linewidth=0.5)
plt.plot(waves, fit_err_plus, color='orange', label=f'Quadratic fit + {percent}$\sigma$', linewidth=0.5)
plt.plot(waves, fit_err_minus, color='orange', label=f'Quadratic fit - {percent}$\sigma$', linewidth=0.5)
plt.plot(waves, fit, color='purple', label='Quadratic fit', linewidth=0.5)
plt.plot(waves, smooth_flux, color='green', label='Smoothened/High Pass Filter')
plt.xlabel('$\lambda [\AA]$')
plt.ylabel('$f_\lambda$ [$erg/cm^2/s/cm$]')
plt.title('Showing Adjusted Starfish Flux From Being Passed Through \n A High-pass Filter And Quadratic Fit Boundaries To Remove Lines')
plt.legend()
plt.show()
plt.plot(waves, adj_flux, color='blue', label='Flux Filter Adjusted', linewidth=0.5) #log
plt.show()

pixels1, autocorrelation = newlagtimes(adj_flux, adj_flux, lags) # Austen's custom Autocorrelation function. 
plt.plot(pixels1, autocorrelation, label="Using Austen's ACF", linewidth=0.5)
plt.title(f'Adjusted (Starfish Flux - Smoothened {high_pass_sigma}$\sigma$) Autocorrelation using Full Spectrum')
plt.ylabel('ACF')
plt.xlabel('Pixels')

# Numpy's autocorrelation method for comparison checks.  
mean = np.mean(adj_flux)
var = np.var(adj_flux)
ndata = adj_flux - mean
acorr = np.correlate(ndata, ndata, 'same')
acorr = acorr /var/ len(ndata)
pixels = np.arange(len(acorr))
if len(acorr)%2 ==0:
    pixels = pixels - len(acorr)/2
else:
    pixels = pixels - len(acorr)/2 +0.5
indx = int(np.where(pixels == 0)[0])
lagpixels = pixels[indx-lags:indx+lags+1]
lagacorr = acorr[indx-lags:indx+lags+1]
plt.plot(lagpixels, lagacorr, color='green', label='Using Numpy Correlate', linewidth=0.5)
plt.legend()
plt.show()

# %% 
"""8 Kernel Calculators"""

# To implement
# %% 
"""8.2 Search grid point indexes for Stage 9"""
search_grid_points = 0 # Make sure to set to 0 if running a remote session!!!
if search_grid_points == 1:
    print("Search different indexes to find the associated grid point values")
    print("Index range is from 0 to {}".format(len(emu.grid_points)-1))
    print("Type '-1' to stop searching")
    print("Increasing the index increases the parameters grid points like an odometer")
    print("--------------------------------------------------------------------------")
    print("Names:", emu.param_names)
    print("Description:", [grid.parameters_description(model_parameters)[i] for i in emu.param_names])
    while True:
        index = int(input("Enter index input"))
        if index == -1:
            break
        elif 0 <= index <= len(emu.grid_points):
            print("Emulator grid point index {}".format(index))
            print(emu.grid_points[index])
        else:
            print("Not valid input! Type integer between 0 and {} or '-1' to quit".format(len(emu.grid_points)-1))

# %%
"""9 Assigning the model and initial model plot"""

### ----- Inputs here ------|
log_amp = -8               # Natural logarithm of the global covariance's Matern 3/2 kernel amplitude log=-52
log_ls = 5                  # Natural logarithm of the global covariance's Matern 3/2 kernel lengthscale
### ------------------------|

model = SpectrumModel(
    'Grid-Emulator_Files/Emulator_full.hdf5',
    data,
    grid_params=list(emu.grid_points[89]), #[list, of , grid , points]
    Av=0,
    global_cov=dict(log_amp=log_amp, log_ls=log_ls)
)
model
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
model.labels

# %% 
"""10 Assigning the priors"""

# Default_priors contains a distribution for every possible parameter
# Mostly uniform across grid space bar global_cov being normal
# Change the default distrubtion if you wish something different. 
default_priors = {
    "param1": st.uniform(1.0e-10,2.9e-9),
    "param2": st.uniform(4, 28), 
    "param3": st.uniform(0.0, 1.0),
    "param4": st.uniform(1.0, 2.0),
    "param5": st.uniform(1e+10, 6e+10),
    "param6": st.uniform(1.0, 5.0),
    "global_cov:log_amp": st.norm(log_amp, 10),
    "global_cov:log_ls": st.uniform(0, 10),
    "Av": st.uniform(0.0, 1.0)
    }

priors = {} # Selects the priors required from the model parameters used
for label in model.labels:
    priors[label] = default_priors[label] # if label in default_priors:
        
# %%
"""11 Training the model"""
model.train(priors)
model
# %%
"""12 Saving and plotting the trained model"""
model.plot(yscale="linear")
model.save("Grid-Emulator_Files/Grid_full_MAP.toml")

# %%
"""12.5 Reloading the trained model"""
model.load("Grid-Emulator_Files/Grid_full_MAP.toml")
model.freeze("global_cov")
model.labels
#%%
#model_ball_initial = {"c1": model["c1"], "c2": model["c2"], "c3": model["c3"]}

# %%
"""13 Set our walkers and dimensionality"""

os.environ["OMP_NUM_THREADS"] = "1"
import multiprocessing as mp
mp.set_start_method('fork', force=True)


### ----- Inputs here ------|
ncpu = cpu_count() - 2      # Pool CPU's used. 
nwalkers = 3 * ncpu         # Number of walkers in the MCMC.
max_n = 1000                # Maximum iterations of the MCMC if convergence is not reached. 
### ------------------------|

ndim = len(model.labels)
print("{0} CPUs".format(ncpu))

default_scales = {"param1": 1e-11, "param2": 1e-2, "param3": 1e-2,
                  "param4": 1e-2, "param5": 1e+9, "param6": 1e-2}

scales = {} # Selects the priors required from the model parameters used
for label in model.labels:
    scales[label] = default_scales[label]
    
# Initialize gaussian ball for starting point of walkers
#scales = {"c1": 1e-10, "c2": 1e-2, "c3": 1e-2, "c4": 1e-2}
#model = model_ball_initial
ball = np.random.randn(nwalkers, ndim)
for i, key in enumerate(model.labels):
    ball[:, i] *= scales[key]
    ball[:, i] += model[key]

# %%
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

backend = emcee.backends.HDFBackend("Grid-Emulator_Files/Grid_full_MCMC_chain.hdf5")
backend.reset(nwalkers, ndim)

with Pool(ncpu) as pool:
    sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_prob, args=(priors,), backend=backend, pool=pool
    )

    index = 0 # Tracking how the average autocorrelation time estimate changes
    autocorr = np.empty(max_n)

    old_tau = np.inf # This will be useful to testing convergence

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

# %% 
"""15 Running extra mcmc steps post-convergence to check clean chains."""
### --- Inputs here ----|
extra_steps = max_n/10  # Extra MCMC steps
### --------------------|
sampler.run_mcmc(backend.get_last_sample(), extra_steps, progress=True)

# %%
"""16 Showing raw MCMC chain."""
reader = emcee.backends.HDFBackend("Grid-Emulator_Files/Grid_full_MCMC_chain.hdf5")
full_data = az.from_emcee(reader, var_names=model.labels)
flatchain = reader.get_chain(flat=True)
print(flatchain)
az.plot_trace(full_data)

# %%
"""17 Discarding MCMC burn-in."""
tau = reader.get_autocorr_time(tol=0)
if m.isnan(tau.max()) == True: 
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

# %%
"""18 Plotting the mcmc chains without the burn-in section,
   summarise our mcmc run's parameters and analysis,
   plot our posteriors of each paramater, 
   produce a corner plot of our parameters."""
az.plot_trace(burn_data);
az.summary(burn_data, round_to=None)
az.plot_posterior(burn_data, [i for i in model.labels]);

# %%
"""19 Producing a corner plot of our parameters."""
# See https://corner.readthedocs.io/en/latest/pages/sigmas/
sigmas = ((1 - np.exp(-0.5)), (1 - np.exp(-2)))
corner.corner(
    burn_samples.reshape((-1, len(model.labels))),
    labels=model.labels,
    quantiles=(0.05, 0.16, 0.84, 0.95),
    levels=sigmas,
    show_titles=True,
)

# %%
"""20 We examine our best fit parameters from the mcmc chains,
   plot and save our final best fit model spectrum."""
ee = [np.mean(burn_samples.T[i]) for i in range(len(burn_samples.T))]
ee = dict(zip(model.labels, ee))
#best_fit = dict(az.summary(burn_data, round_to=None)["mean"])
model.set_param_dict(ee)
model
model.plot();
model.save("Grid-Emulator_Files/Grid_full_parameters_sampled.toml")

# %%
