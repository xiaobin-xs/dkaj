# Deep Kernel Aalen–Johansen (DKAJ) Estimator
Code for the paper:
> Xiaobin Shen* and George H. Chen*. "Deep Kernel Aalen–Johansen Estimator: An Interpretable and Flexible Neural Net Framework for Competing Risks." *Machine Learning for Health* 2025. [[arXiv](https://arxiv.org/abs/2512.08063)]

🏆 Best Paper Award (Models and Methods) @ ML4H December, 2025

DKAJ is an interpretable competing-risks model that generalizes the classical Aalen–Johansen (AJ) estimator to produce individual-level cumulative incidence functions (CIFs) via a learned kernel / representation.

## Key idea (interpretability)
For an input feature vector `x`, DKAJ represents `x` as a **weighted combination of clusters** of training points. In an extreme case where `x` puts weight on only one cluster, the predicted CIFs correspond to the classical AJ estimator computed on that cluster. (See paper for details.)

## What’s in this repo
This repository contains:
- DKAJ training + prediction code
- Baselines used in the paper:
  - Fine-Gray (FG)
  - Cause-specific Cox (cs-Cox)
  - Random Survival Forest w/ Competing Risks (RSF-CR)
  - DeepHit
  - Deep Survival Machines (DSM)
  - Neural Fine-Gray (NeuralFG)
  - SurvivalBoost
- Experiment runners and configuration files for reproducing paper results


## Project Structure

```
.
├── dkaj_train_demo.ipynb           # Interactive tutorial
├── datasets.py                     # Dataset loading and preprocessing
├── models.py                       # Model implementations (DKAJ, baselines, etc.)
├── modelsR.py                      # Utilities for models implemented in R
├── metrics.py                      # Evaluation metrics
├── visualization_utils.py          # Plotting utilities
├── config/                         # Config files directory
│   ├── dkaj.ini                        # DKAJ config
│   ├── baseline.ini                    # Baselines config
│   ├── dkaj_train_size.ini             # DKAJ config with varying training set sizes
│   ├── baseline_train_size.ini         # Baselines config with varying training set sizes
├── data/                           # Data directory
│   ├── framingham.csv                  # Framingham dataset
│   ├── pbc2.csv                        # PBC dataset
├── experiments/                    # Experiment runners
│   ├── run_dkaj.py                     # Main DKAJ experiment
│   ├── run_dkaj_no_loo.py              # DKAJ without LOO loss
│   ├── run_dkaj_no_tuna.py             # DKAJ without TUNA warm-up
│   ├── run_csCox.py                    # Cause-specific Cox baseline
│   ├── run_FG.py                       # Fine–Gray baseline
│   ├── run_rsfcr.py                    # RSF-CR baseline
│   ├── run_deephit.py                  # DeepHit baseline
│   ├── run_dsm.py                      # Deep Survival Machines baseline
│   ├── run_neuralFG.py                 # Neural Fine–Gray baseline
│   └── run_survboost.py                # SurvivalBoost baseline
├── dsm/                            # Deep Survival Machines implementation
├── nfg/                            # Neural Fine–Gray implementation
├── pycox/                          # PyCox moduel with minor modification
└── torchtuples/                    # torchtuples moduel with minor modification
```



## Installation

### Requirements
- Python 3.7+
- PyTorch
- scikit-learn
- lifelines
- hazardous
- hnswlib

### Setup

```bash
pip install -r requirements.txt
```

### Third-party code
This repo vendors code from the following sources:
- `dsm/` and `nfg/`: Deep Survival Machines and Neural Fine-Gray implementations referenced from [here](https://github.com/Jeanselme/DeepSurvivalMachines/tree/e4b07b3f497f2266eaa71d0e182195e95663d367/dsm) and [here](https://github.com/Jeanselme/NeuralFineGray/tree/main/nfg) 
- `pycox/` and `torchtuples/`: slightly modified versions of PyCox and torchtuples borrowed [here](https://github.com/georgehc/survival-kernets)

Please see the corresponding directories for original licenses and attribution.

## Quick Start

### 1. Demo Notebook

A demo notebook (training + basic evaluation) is provided here:
- `dkaj_train_demo.ipynb`

The notebook demonstrates the complete DKAJ pipeline on the Framingham dataset:
1. Load and preprocess data
2. Fit SurvivalBoost (base model)
3. TUNA warm-up of a base neural network using SurvivalBoost leaves
4. Train DKAJ on top of the warm-started network
5. Optional summary fine-tuning
6. Evaluation using time-dependent concordance (C-td) and integrated Brier score (IBS)

- [ ] ToDo: add demo for visualization


### 2. Reproducing experiments (paper)

Experiment scripts read hyperparameters and dataset settings from config files.

#### Full DKAJ Pipeline

```bash
python experiments/run_dkaj.py config/dkaj.ini
```

#### Baseline Methods

Run individual baseline methods with standard configuration:

```bash
python experiments/run_csCox.py config/baseline.ini
python experiments/run_FG.py config/baseline.ini
python experiments/run_rsfcr.py config/baseline.ini
python experiments/run_deephit.py config/baseline.ini
python experiments/run_dsm.py config/baseline.ini
python experiments/run_neuralFG.py config/baseline.ini
python experiments/run_survboost.py config/baseline.ini
```

#### Experiments with Varying Training Size

To evaluate methods on datasets with different training set sizes:

```bash
python experiments/run_dkaj.py config/dkaj_train_size.ini
python experiments/run_csCox.py config/baseline_train_size.ini
python experiments/run_FG.py config/baseline_train_size.ini
python experiments/run_rsfcr.py config/baseline_train_size.ini
python experiments/run_deephit.py config/baseline_train_size.ini
python experiments/run_dsm.py config/baseline_train_size.ini
python experiments/run_neuralFG.py config/baseline_train_size.ini
python experiments/run_survboost.py config/baseline_train_size.ini
```



#### Outputs

Runs write metrics (e.g., IBS, time-dependent concordance) and predictions to an output directory configured in the INI files.


### 3. Ablation Studies

Test DKAJ without specific components:

#### Without Leave-One-Out Loss

```bash
python experiments/run_dkaj_no_loo.py config/dkaj.ini
```

This variant removes the leave-one-out adjustment to evaluate its contribution.

#### Without TUNA Warm-up

```bash
python experiments/run_dkaj_no_tuna.py config/dkaj.ini
```

This variant trains the neural network from scratch without TUNA initialization.

## Datasets

The paper evaluates on four standard competing-risks datasets (each with two event types):
- PBC
- Framingham
- SEER (subset restricted to diagnoses in 2010; see the paper for details.)
- Synthetic (Lee et al., 2018).

The repository includes PBC and Framingham datasets in the `data/` directory. 
SEER is publicly available, but requires a registration process to obtain access; therefore we do not redistribute SEER data in this repository. 
For Synthetic dataset, the code reads data from the original [DeepHit repository](https://github.com/chl8856/DeepHit/tree/master/sample%20data/SYNTHETIC).

See `datasets.py` for details on how data is loaded and preprocessed.


## Evaluation metrics

We report:
- Time-dependent concordance index (C-td)
- Integrated Brier score (IBS) for competing risks



## Citation

If you use this code, please cite:
```
@inproceedings{shen2025dkaj,
  title     = {Deep Kernel Aalen--Johansen Estimator: An Interpretable and Flexible Neural Net Framework for Competing Risks},
  author    = {Shen, Xiaobin and Chen, George H.},
  booktitle = {Machine Learning for Health (ML4H)},
  year      = {2025},
  note      = {*Equal contribution.}
}
```