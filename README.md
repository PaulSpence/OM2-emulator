# OM2-emulator
Overall, the aim here is learn how to create a machine learning emulator for access-om2 models. The hope is that an emulator will allow us to generate ensembles of simulated output based on a high resolution model data, at a much cheaper cost than running an ensemble of the model itself. The primary benefit being that it will allow us to better distinguish internal from forced variability in our model simulation results. 

## Aim 0:
Create latent space for vertically integrated ocean heat content and net surface heat fluxes. Follow the pyearth tools autoencoder_example tutorial. See here for details: https://github.com/PaulSpence/OM2-emulator/issues/7

## Aim 1: 
Emulate SST from ACCESS-CM2 (using SAT and wind stress as inputs). Essentially reproduce some results from Dheeshjith et al. 2024 (https://arxiv.org/abs/2405.18585). See here for details: https://github.com/PaulSpence/OM2-emulator/issues/1#issue-2535235521
Regrid: 1 deg om2, 1 deg global since om2 has 1/3deg near the equator and 1 deg at poles to resolve the undercurrents.


## Aim 2: 
Redo Aim 1, but using ACCESS-OM2-01 ocean data and future atmosphere from Qian or Hannahs future warming runs. See here: https://github.com/PaulSpence/OM2-emulator/issues/2#issue-2535251725

## Aim 3: 
Since emulating SST from SAT doesn't seem that challenging, we would like to try to autoregressively emulate ACCESS-OM2-1’s vertically integrated ocean heat content evolution given surface forcing (basically, emulate Huguenin et al. 2022; https://www.nature.com/articles/s41467-022-32540-5 Nat Comms.) See here: https://github.com/PaulSpence/OM2-emulator/issues/3#issue-2535255067

# Ryan's process for getting PyEarthTools working (latest) and notebooks working on GPU hopper

### ARE session settings:
```
Cluster: ncigadi
Walltime (hours): 2
Queue: gpuhopper
Compute Size: 1gpu
Project: nm47
Storage: gdata/nm47+gdata/dk92+gdata/dx2
Software:
Settings:
Show advanced settings: 1
Extra arguments:
Module directories: /g/data/dk92/apps/Modules/modulefiles/
Modules: pet/2025.08
Python or Conda virtual environment base:
Conda environment:
Environment variables: PYTHONUSERBASE=/g/data/dx2/rmh561/python-userbase
Jobfs size: 100GB
```

### Installing PyEarthTools:
- clone latest `develop` branch from https://github.com/ACCESS-Community-Hub/PyEarthTools to somewhere on gdata.
- In that PyEarthTools directory, ensure that
```
pet > echo $PYTHONUSERBASE
/g/data/dx2/rmh561/python-userbase
```
- Then do `/opt/conda/envs/pet/bin/python -m pip install --user -r requirements.txt`
- Once installed, open a Jupyter notebook with the "PET-Python" kernel.
- Run
```
import pyearthtools
print(pyearthtools.__path__)
```
If this is pointing to directories in your PyEarthTools repository on gdata, then you're good to go.
