TODO: add a demo notebook.

## Main experiments - Baseline and DKAJ
To run experiment, under main directory, run, for example:
```shell
python experiments/run_csCox.py config.ini
python experiments/run_deephit.py config.ini
python experiments/run_dsm.py config.ini
python experiments/run_FG.py config.ini
python experiments/run_neuralFG.py config.ini
python experiments/run_survboost.py config.ini

python experiments/run_dkaj.py config_dkaj.ini
```

## Experiments with varying training size
Run experiments with different training size on the synthetic dataset:
```shell
python experiments/run_FG.py config_train_size.ini
python experiments/run_csCox.py config_train_size.ini
python experiments/run_deephit.py config_train_size.ini
python experiments/run_dsm.py config_train_size.ini
python experiments/run_neuralFG.py config_train_size.ini
python experiments/run_survboost.py config_train_size.ini
python experiments/run_dkaj.py config_dkaj_train_size.ini
```

## Abalation study
Run experiments with no leave-one-out loss
```shell
python experiments/run_dkaj_no_loo.py config_dkaj.ini
```

Run experiments with no TUNA warm-up
```shell
python experiments/run_dkaj_no_tuna.py config_dkaj.ini
```