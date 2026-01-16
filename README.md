# OM2-emulator
Overall, the aim here is learn how to create a machine learning emulator for access-om2 models. The hope is that an emulator will allow us to generate ensembles of simulated output based on a high resolution model data, at a much cheaper cost than running an ensemble of the model itself. The primary benefit being that it will allow us to better distinguish internal from forced variability in our model simulation results. 

## Aim 0:
Put in 2D net surface heat fluxes and verticall integrated 2d ocean heat content in at t=0, get there latent space representation, from om2. 
Get monthly averages from 1 deg IAF run. cycle 6 of the omip2 run. /g/data/ik11/outputs/access-om2/1deg_jra55_iaf_omip2_cycle6
vars needed: net_surface_heating+frazil_int_3d, compute vertically integrated heat content, temp*rho_dzt*cp and sum vertically

Surface flux: net_sfc_heating + frazil_3d_int_z (see https://github.com/COSIMA/access-om2/issues/139#issuecomment-639278547)

Vertically integrated heat content:
temp*rho_dzt*Cp

Cp = 3992.10322329649
area_t

## Aim 1: 
Emulate SST from ACCESS-CM2 (using SAT and wind stress as inputs). Essentially reproduce some results from Dheeshjith et al. 2024 (https://arxiv.org/abs/2405.18585). See here for details: https://github.com/PaulSpence/OM2-emulator/issues/1#issue-2535235521
Regrid: 1 deg om2, 1 deg global since om2 has 1/3deg near the equator and 1 deg at poles to resolve the undercurrents.


## Aim 2: 
Redo Aim 1, but using ACCESS-OM2-01 ocean data and future atmosphere from Qian or Hannahs future warming runs. See here: https://github.com/PaulSpence/OM2-emulator/issues/2#issue-2535251725

## Aim 3: 
Since emulating SST from SAT doesn't seem that challenging, we would like to try to autoregressively emulate ACCESS-OM2-1’s vertically integrated ocean heat content evolution given surface forcing (basically, emulate Huguenin et al. 2022; https://www.nature.com/articles/s41467-022-32540-5 Nat Comms.) See here: https://github.com/PaulSpence/OM2-emulator/issues/3#issue-2535255067
