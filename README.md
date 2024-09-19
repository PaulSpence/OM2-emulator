# OM2-emulator
Overall, the aim here is learn how to create a machine learning emulator for access-om2 models. The hope is that an emumlator will allow to generate ensembles of simulated output based on a high resolution model data. The primary benefit being that it will allow us to better distinguish internal from forced varaiability in our model simulation results. 

## Aim 1: 
Emulate SST from SAT using data from ACCESS-CM2, essentially to reproduce some results from Dheeshjith et al. 2024 (https://arxiv.org/abs/2405.18585). See here for details: https://github.com/PaulSpence/OM2-emulator/issues/1#issue-2535235521

## Aim 2: 
Redo Aim 1, but using ACCESS-OM2-1 ocean data and future atmosphere from Qian or Hannahs future warming runs.

## Aim 3: 
Since emulating SST from SAT doesn't seem that challenging, we would like to try to autoregressively emulate ACCESS-OM2-1’s vertically integrated ocean heat content evolution given surface forcing (basically, emulate Huguenin et al. 2022; https://www.nature.com/articles/s41467-022-32540-5 Nat Comms.)
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
