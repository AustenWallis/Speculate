# Speculate functions

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import itertools

def plot_emulator(emulator, grid, not_fixed, fixed):
    
    """Takes the emulator and given the emulator's inputed model parameters, 
    displays the weights and Gaussian process interpolation from the SVD (PCA decomposition).

    Args:
        emulator (Starfish.emulator.emulator): Trained emulator from your grid space
        
        removed: model_parameters (tuple): The numbers corresponding to the modelling parameters of the grid
        
        not_fixed (int): The varying model parameter number that the weights plot displays (x-axis)
        
        fixed (int): The python list index of the other model parameters. 
            If a model parameter has 5 values with the grid space. 
            Possible int ranges would be 0-4.
        
    Returns:
        Plot of PCA component weights.
    """
    
    # Placing the grid points values within a dictionary, keyed as 'params{}'
    variables = {}
    for loop in grid.model_parameters:
        variables["param{}".format(loop)] = np.unique(emulator.grid_points[:, grid.model_parameters.index(loop)])
        
    # Creating a custom itertools.product routine which can dynamically input the free varying parameter
    # and the length of the number of parameters depending on what is specified. 
    # params = np.array(list(itertools.product(T, logg[:1], Z[:1]))) # <-- starfish original
    not_fixed_index = grid.model_parameters.index(not_fixed) # Converting parameter number to index position
    params = []
    temp = [variables[emulator.param_names[j]] for j in range(len(variables))] # Creating list from dictionary
    temp2 = [np.array(temp[i]) if i==not_fixed_index else temp[i][fixed] for i in range(len(temp))] # New list fixing the other parameters on the given grid point
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
    
    param_x_axis = np.unique(emulator.grid_points[:,not_fixed_index]) # Picking out all unique not fixed parameter values
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
    xlabel = grid.parameters_description()[f"param{not_fixed}"]
    for i, (m, s) in enumerate(zip(mus.T, sigs.T)):
        axes[i].plot(param_x_axis_test, m, "C1")
        axes[i].fill_between(param_x_axis_test, m - (2 * s), m + (2 * s), color="C1", alpha = 0.4)
        axes[i].set_xlabel(f"Parameter $log_{{10}}({xlabel})$")
    plt.suptitle(f"Weights for Parameter {xlabel} with the other parameters fixed to their {fixed} index grid point", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
def grid_spectrum_plot(grid, wl_range):
    print('hi')

def emulator_spectrum_plot():
    print('hi')

def simplex(model, priors):
    """An initial simplex algorithm for training the model's Nelder_Mead routine. 
    Suitable for normal or uniform distributions only. If different add method to simplex routine. 
    We divide the each parameter's range into intervals with equal spacing, equal to the 
    number of parameters. Each simplex column point's, (num of parameters+1) in total, contains interval values
    of an individual parameter's range. Each simplex column (parameter values) is then cycled + 1 compared to each 
    other with np.roll. i.e a rolling simplex (see numpy.roll).
     
    Args:
        model (Starfish.models.spectrum_model.SpectrumModel): A starfish model class with emulator and data
        set built in.
        priors (dictionary): Parameter priors for the minimisation training and the MCMC. 

    Returns:
        simplex: An initial simplex to use for the nelder_mead routine when training the model class with scipy.optimise.minimise.
    """
    
    def uniform_prior(distribution, N):
        "Method for generating simplex values for a uniform distribution prior parameter"
        min_value = distribution.args[0]
        max_value = distribution.args[1] + distribution.args[0]
        the_range = max_value - min_value
        truncated_perc = the_range / 20 # reducing by 20% for both ends of the range 
        truncated_min = min_value + truncated_perc
        truncated_max = max_value - truncated_perc
        truncated_range = truncated_max - truncated_min
        interval = truncated_range / N
        data_column = [truncated_min + interval * multiple for multiple in range(N+1)] # even value distribution across range
        return data_column
    
    def normal_prior(distribution, N):
        "Method for generating simplex values for a noramlly prior parameter"
        mean = distribution.args[0]
        std = distribution.args[1]
        min_value = mean - (std*2) # +/- 2 standard deviations for the range
        max_value = mean + (std*2)
        the_range = max_value - min_value
        interval = the_range / N
        data_column = [min_value + interval * multiple for multiple in range(N+1)] # even value distribution across range
        return data_column
    
    N = len(model.get_param_vector()) # Number of training parameters
    simplex = np.zeros((N+1, N)) # Required size of the simplex for given number of parameters
    iteration = 0 # loop count to add to the simplex columns
    for name, distribution in priors.items(): # priors being used
        if isinstance(distribution.dist, type(st.uniform)): # check if distribution is uniform
            data_column = uniform_prior(distribution, N) # returns an even data vector/'column'
        elif isinstance(distribution.dist, type(st.norm)): # check if distribution is normal
            data_column = normal_prior(distribution, N) # returns an even data vector/'column'
        else:
            raise(f"Error: {name} is {type(distribution.dist)} and not a uniform or normal distribution") 
        simplex[:,iteration] = data_column # adding data column to simplex
        iteration += 1 # next loop

    # rolling the simplex columns +1 each additional column
    for column in range(N):
        new_column = np.roll(simplex[:,column], column)
        simplex[:,column] = new_column

    return simplex

def is_matrix_pos_def(square_matrix):
    """Simple check to see if square matrix is positive definite. 
        If all eigenvalues are greater than 0. Then true is returned.
        For semi-definite, set to greater than or equal to 0

    Returns:
        boolean: True if positive definite matrix, False if not.
    """
    return np.all(np.linalg.eigvals(square_matrix) > 0)


def unique_grid_combinations(grid):
    """Method to return the unique combinations of grid points in a list.
    
    Returns:
        list: A list of the unique combinations of grid points.
    """
    unique_combinations = []
    for i in itertools.product(*grid.points):
        unique_combinations.append(list(i))
    
    return unique_combinations

def plot_grid_point():
    pass

def search_grid_points(switch, emu, grid):
    if switch == 1:
        print("- Search index range 0 to {} to find the associated grid point values.".format(len(emu.grid_points) - 1))
        print("- Type '-1' to stop searching.")
        print("- Or type [Input, Parameter, Values] in a list for the specific grid point.")
        print("- No square brackets needed, just commas.")
        print("- Increasing the index increases the parameters grid points like an odometer.")
        print("---------------------------------------------------------------")
        print("Names:", emu.param_names)
        print("Description:", [grid.parameters_description()[i] for i in emu.param_names])
        while True:
            user_input = input("Enter Index Value or {} Parameter Values Separated By Commas".format(
                len(emu.param_names)))
            user_input = user_input.split(",") # turning string input into list
            print("-----------------------------------------------------------")
            if len(user_input) == 1:
                index = int(user_input[0]) # variable for integer index
                grid_vals = None # resetting previous variables for other scenarios.
            else:
                grid_vals = [float(i) for i in user_input] # for parameter list
                index = None
            
            # different senarios for user inputs, index or parameter values    
            grid_points = list(emu.grid_points) # fix for enumerating 
            if index == -1: # quit
                break
            
            # integer grid point search
            elif isinstance(index, int) and 0 <= index <= len(emu.grid_points):
                better_display = [grid_points[index][i] for i in range(len(grid_points[index]))]
                print("Emulator grid point index {} is".format(index), 
                      better_display)
            
            # parameter grid point search        
            elif isinstance(grid_vals, list) and len(grid_vals) == len(emu.param_names):
                skipover = True # A variable to print if point not in emulator
                for index, points in enumerate(grid_points): # enumerating to get index
                    # if there is a grid point that matches the input
                    if all(np.round(points,3) == np.round(grid_vals,3)):
                        print("Parameter's {} is at emulator grid point index {}".format(grid_vals, index))
                        skipover = False # to avoid printing the skipover message
                        break # stop searching enumeration
                if skipover:
                    print("Grid point {} is not in the emulator".format(grid_vals))
                    
            # invalid input, can't find grid point
            else:
                print("{} Not valid input! \nType integer between 0 and {}".format(
                    user_input, len(emu.grid_points) - 1) + 
                    ", a parameter list of length {}".format(
                        len(emu.grid_points[0])) + " or '-1' to quit")
