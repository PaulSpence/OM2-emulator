# OM2-emulator
Overall, the aim here is learn how to create a machine learning emulator for access-om2 models. The hope is that an emumlator will allow to generate ensembles of simulated output based on a high resolution model data. The primary benefit being that it will allow us to better distinguish internal from forced varaiability in our model simulation results. 

## Aim 1: 
First simple aim is to reproduce some results from Dheeshjith et al. 2024 (https://arxiv.org/abs/2405.18585) except using for ACCESS-CM2. 

Dheeshjith et al. used GFDL-CM2.6 data (PI, 2xCO2, and 4xCO2). They used 4000 daily data samples from the 20 year PI control run to train the emulators. They test on the PI and 2xCO2 runs using an initial state fromd ay 4200 and atmospheric boundary info through day 7200. They test on the 4xCO2 data.

Their goal is to autoregressively emulate the surface ocean climate given the atmospheric boundary conditions, and test the generalization to different atmospheric boundary conditions (e.g.increased CO2). 
They define the ocean state as phi=(u,v,T,t): zonal and meridional surface velocity and SST. 
They define the atmospheric boundary condition as tau=(tx,ty, Tatm,t): zonal, meridional and SAT.

Tey predict phi(t+delta_t) using input from phi(t) and tau(t) with delta_t = 1 day. Apparently this gives 6 input channels (u,v,T,tx,ty,Tatm at time=t) and 3 output (u,v,T) channels (?). 

## Aim 2: 
Do the same for ACCESS-OM2-01, aiming to emulate Qian’s runs (all coarse-grained to 1-degree).

## Aim 3: 
autoregressively emulate ACCESS-OM2-1’s vertically integrated ocean heat content evolution given surface forcing (basically, emulate Huguenin et al. Nat Comms.)
Ocean state: OHC(x,y,t) [SST(x,y,t)]
Atmospheric forcing: U, V, Ta, (q, radiation) (x,y).
Model: inputs: OHC(x,y,t), U,V Ta, q(x,y,t) -> output: OHC*(x,y,t+delta t).

Possible training/testing dataset:
OMIP-2 ACCESS-OM2-1 IAF run /// Maurice’s historial
RDF /// Mauric’e historical
Early part of Maurice’s historial /// late part of Maurice’s historical 
All of Maurice’s run /// wind and thermal forcing
All of Maurice’s run /// Qian’s run (or something like it)
Metrics:
Global OHC
Spatial structure of OHC (SO average…)
