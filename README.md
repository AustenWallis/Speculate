# Speculate

## The spectral emulator built for 'python', in python...

Designed for the rapid testing of an observational spectrum to determine the physical properties of the astronomical object. 

More information to be added later. Have a look at the cookbook notebook for descriptions of the code for now.  

### Files description:
- **Grid-Emulator_Files** - The location where your grid/emulator.hdf5 and other saved files are placed when running. This will change every time I push a new commit as it's the dumping ground of data.
- **kgrid** - This stores the ss-cygni test grid we are testing from. You can place any cv data spectrum file in here from python you wish like 'ss_cyg_1.spec'.
- **Speculate.py** - Dev version of speculate. I run it in VS Code with the anaconda environment. With Jupyter extenstions, the `# %%` allows the code to be run like a notebook. Alternatively, you could run the file but you'll be unable to pause the code at certain points. 
- **Speculate Cookbook Guide.ipynb** - The stable version of speculate. The python notebook contains information and explanations of the working code. 
- **sscyg_k2.ls** - Metadata of the kgrid (file names corresponding to parameters/variable names.)
- **.gitattributes** and **README.md** - GitHub features. 


#### To do list (21st April 2023):
- Find ways manipulating the data to reduce correlations between alternative grid points. (Hopefully solve uniform chains and cornerplots.) Check with starfish demo too. 
- Interpolated data in stage 6 switch 3, check noisy data switch 2 to smoothen or not.
- Check gaussian ball inital starting points to be tight.
~ Log/scale/linear(unchanged) switch to change the domains the observations and emulator work in. Which is best for the data. 
- Log_amp/Log_ls global kernel calculators for estimate for stage 9. 
- Log-probability limits for the variable parameters. 

#### 25.04.2023 Updates

Correlation grid points matrix debugging.
MCMC completing. 
Updated cookbook.
Custom Starfish folder for install required (read cookbook install instructions)
Scaling switch semi-implemented for log/scale/linear. Very small effects in outcomes. 'Log' faster but less sensitive. 