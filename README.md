# OM2-emulator

Overall, the aim here is learn how to create a machine learning emulator for access-om2 models. The hope is that an emulator will allow us to generate ensembles of simulated output based on a high resolution model data, at a much cheaper cost than running an ensemble of the model itself. The primary benefit being that it will allow us to better distinguish internal from forced variability in our model simulation results.

## Refactored Experiment Workflow

This repository now supports a reproducible **CLI + YAML** experiment workflow while still using the existing **PyEarthTools** data path and `PartialConv2d` model family.

### Why this structure

This is the coding standard we want for iterative scientific ML analysis because it provides:
- reproducibility (every run writes a resolved config and metrics),
- comparability (same training/evaluation pipeline across variants),
- traceability (run artifacts include diagnostics and summary tables),
- faster iteration (hyperparameters changed in YAML rather than notebook code).

### Environment setup (NCI)

Load analysis3 before running the new scripts/tests:

```bash
module load conda/analysis3
```

### New entrypoints

- `scripts/run_experiment.py`
  - Runs one config.
- `scripts/run_ablation.py`
  - Runs baseline + one-factor-at-a-time ablation variants.

### Config files

- `configs/baseline.yaml`
  - Baseline architecture, normalization, training, and diagnostics setup.
- `configs/ablation_matrix.yaml`
  - Factor list for one-factor-at-a-time sweeps.

### Typical usage

Run one experiment:

```bash
module load conda/analysis3
python scripts/run_experiment.py --config configs/baseline.yaml
```

Run ablation matrix:

```bash
module load conda/analysis3
python scripts/run_ablation.py \
  --baseline configs/baseline.yaml \
  --matrix configs/ablation_matrix.yaml
```

### Notebook diagnostics

A notebook dashboard is provided at:
- `notebooks/Ablation_Diagnostics.ipynb`

It reads `runs/summary.csv` and renders:
- ranked run tables,
- metric bar plots,
- side-by-side reconstruction and latent diagnostic panels.

### Tests

Run the regression tests with:

```bash
module load conda/analysis3
python -m unittest discover -s tests -p 'test_*.py' -v
```

### Output structure

Each run is written to `runs/<run_name>_<timestamp>/` and includes:
- `resolved_config.yaml`
- `metrics.json`
- `training_summary.json`
- `model_state.pt`
- `arrays.npz` (optional)
- `diagnostics/reconstruction_maps.png`
- `diagnostics/latent_feature_maps.png`

A cross-run comparison table is appended to:
- `runs/summary.csv`

## Aim 0

Create latent space for vertically integrated ocean heat content and net surface heat fluxes. Follow the pyearth tools autoencoder_example tutorial. See here for details: https://github.com/PaulSpence/OM2-emulator/issues/7

## Aim 1

Emulate SST from ACCESS-CM2 (using SAT and wind stress as inputs). Essentially reproduce some results from Dheeshjith et al. 2024 (https://arxiv.org/abs/2405.18585). See here for details: https://github.com/PaulSpence/OM2-emulator/issues/1#issue-2535235521
Regrid: 1 deg om2, 1 deg global since om2 has 1/3deg near the equator and 1 deg at poles to resolve the undercurrents.

## Aim 2

Redo Aim 1, but using ACCESS-OM2-01 ocean data and future atmosphere from Qian or Hannahs future warming runs. See here: https://github.com/PaulSpence/OM2-emulator/issues/2#issue-2535251725

## Aim 3

Since emulating SST from SAT doesn't seem that challenging, we would like to try to autoregressively emulate ACCESS-OM2-1’s vertically integrated ocean heat content evolution given surface forcing (basically, emulate Huguenin et al. 2022; https://www.nature.com/articles/s41467-022-32540-5 Nat Comms.) See here: https://github.com/PaulSpence/OM2-emulator/issues/3#issue-2535255067
