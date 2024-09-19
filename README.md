# OM2-emulator

## Aim 1: 
Do Dheeshjith et al. 2024 except for ACCESS-CM2.

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
