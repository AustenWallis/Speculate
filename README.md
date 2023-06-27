# Speculate

## The spectral emulator built for 'python', in python...

Designed for the rapid testing of an observational spectrum to determine the physical properties of the astronomical object. 

More information to be added later. Have a look at the cookbook notebook for descriptions of the code for now.  

#### <u>**Installation**</u>

To run, simply download the entire Speculate main branch, install package dependencies and run the cookbook. The download is large due to two spectrum grid's (kgrid and short_spec_cv_grid).

### Files description:
The speculate directory contains a number of items:
- <u>**Speculate Cookbook Guide.ipynb**</u> - This is the stable version of speculate. The python notebook contains information and explanations of the working code for you to use. This is the easiest way to run. 
- <u>**Speculate_dev.py**</u> - Dev version of speculate. This version is likely the most current iteration but may not run correctly. I run it in VS Code utilising a mix of intel/m1 anaconda environment's. With Jupyter extenstions, the `# %%` allows the code to be run like a notebook. Alternatively, you could simply run the file but you'll be unable to pause the code at certain points, which again may break the code if certain 0/1 switches in the code are incorrectly on/off.
- <u>**kgrid**</u> - This stores a ss-cygni cv test grid you can create an emulator from. You can place any cv data spectrum file from 'python' as a data/observation source in this directory. Example, like 'ss_cyg_1.spec'. Speculate can then attempt to find the parameters for this data/observation file (see stage 6). **sscyg_k2.ls** - Metadata of the kgrid (file names corresponding to parameters/variable names.)
- <u>**short_spec_cv_grid**</u> - This stores a default cv blackbody python v87a grid space of 3 parameters. For reference, a template.pf 'python' file is here that the grid was made from.
- <u>**plots**</u> - This is a dumping folder to place any saved plots in the code.
- <u>**Grid-Emulator_Files**</u> - This is where your saved grid, emulator and MCMC files are saved. Ensure your file names are systematic and explicit so that you can retrieve previous trained emulators and mcmc runs. Otherwise, you may rerun and overwrite accidentally. 
- <u>**Speculate_addons**</u> - This stores the additional import modules such as grid interfaces for Speculate's grids and other functions.
- <u>**Starfish**</u> - This is the original Starfish program. This stores the import module necessary for the program to run.
- <u>**.gitattributes** and **README.md**</u> - GitHub features. 



For my installation, I'm using a conda environment with python 3.9.13 and Starfish version 0.4.2 (most up to date version 4/4/23). The Starfish package is located within speculate's 'Starfish' directory and hopefully the code should look through your relative path to find starfish's classes/functions. You shouldn't separately need to install Starfish. You can see the Starfish Github here: https://github.com/Starfish-develop/Starfish. 

However, you are needed to install a suite of extra packages if you don't already have them. Not all might be listed here. 
- emcee ($conda install -c conda-forge emcee)
- arviz ($conda install -c conda-forge arviz)
- corner($conda install -c astropy corner)
- **NDtyping, ensure version 1.4.4** ($pip install nptyping==1.4.4), later versions cause issues I have yet to diagnose.

Keep doing $pip install or $conda installing until importing in Stage 1 stops yelling code errors at you. 

Once downloaded, installed and Stage 1 completes, you should be set! 