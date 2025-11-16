#!/usr/bin/env python
import ast
import configparser
import csv
import gc
import hashlib
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import pickle
import random
import shutil
import sys
sys.path.insert(1, os.path.dirname(sys.path[0]))
import time
import uuid

import numpy as np
import pandas as pd
import torch
# torch.set_deterministic(True)  # causes problems with survival analysis
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
from sklearn.model_selection import train_test_split
from lifelines import KaplanMeierFitter

import torchtuples as tt

from datasets import load_dataset, LabTransformCR
from models import CauseSpecificNet, DeepHit
from metrics import neg_cindex_td, c_index_competing_single_time, compute_brier_competing_multiple_times, compute_ibs_competing


estimator_name = 'deephit'

if not (len(sys.argv) == 2 and os.path.isfile(sys.argv[1])):
    print('Usage: python "%s" [config file]' % sys.argv[0])
    sys.exit(1)

config = configparser.ConfigParser()
config.read(sys.argv[1])

n_experiment_repeats = int(config['DEFAULT']['n_experiment_repeats'])
fix_test_shuffle_train = \
    int(config['DEFAULT']['fix_test_shuffle_train']) > 0
val_ratio = float(config['DEFAULT']['simple_data_splitting_val_ratio'])
datasets = ast.literal_eval(config['DEFAULT']['datasets'])
output_dir = config['DEFAULT']['output_dir']
method_header = 'method: %s' % estimator_name
method_random_seed = int(config[method_header]['random_seed'])
patience = int(config[method_header]['early_stopping_patience'])
eval_horizon_quantiles = ast.literal_eval(config['DEFAULT']['eval_horizon_quantiles'])
ibs_n_horizon_points = int(config['DEFAULT']['ibs_n_horizon_points'])
ibs_max_horizon_percentile = float(config['DEFAULT']['ibs_max_horizon_percentile'])
model_selection_metric = config['DEFAULT']['model_selection_metric'][:6]
verbose = int(config['DEFAULT']['verbose']) > 0

assert model_selection_metric in ['avgIBS', 'avgCtd'],\
    f"model selection metric must be either 'avgIBS' or 'avgCtd', got {model_selection_metric}"

compute_bootstrap_CI = int(config['DEFAULT']['compute_bootstrap_CI']) > 0
bootstrap_CI_coverage = float(config['DEFAULT']['bootstrap_CI_coverage'])
bootstrap_n_samples = int(config['DEFAULT']['bootstrap_n_samples'])
bootstrap_random_seed = int(config['DEFAULT']['bootstrap_random_seed'])

if model_selection_metric == 'avgCtd':
    output_dir += f'_{model_selection_metric}'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)

max_n_epochs = int(config[method_header]['max_n_epochs'])

sigma_range = ast.literal_eval(config[method_header]['sigma'])
hyperparams = \
    [(alpha, sigma, n_durations, n_layers, n_nodes, batch_size, max_n_epochs,
      lr)
     for alpha
     in ast.literal_eval(config[method_header]['alpha'])
     for sigma
     in sigma_range
     for n_durations
     in ast.literal_eval(config[method_header]['n_durations'])
     for n_layers
     in ast.literal_eval(config[method_header]['n_layers'])
     for n_nodes
     in ast.literal_eval(config[method_header]['n_nodes'])
     for batch_size
     in ast.literal_eval(config[method_header]['batch_size'])
     for lr
     in ast.literal_eval(config[method_header]['learning_rate'])]

hyperparam_hash = hashlib.sha256()
hyperparam_hash.update(str(hyperparams).encode('utf-8'))
hyperparam_hash = hyperparam_hash.hexdigest()

validation_string = 'vr%f' % val_ratio

output_test_table_filename \
    = os.path.join(output_dir,
                   '%s_experiments%d_%s_test_metrics_%s.csv'
                   % (estimator_name,
                      n_experiment_repeats,
                      validation_string,
                      hyperparam_hash))
output_test_table_file = open(output_test_table_filename, 'w')
test_csv_writer = csv.writer(output_test_table_file)
if compute_bootstrap_CI:
    test_csv_writer.writerow(['dataset',
                              'experiment_idx',
                              'method',
                              'loss',
                              'loss_CI_lower',
                              'loss_CI_upper'])
else:
    test_csv_writer.writerow(['dataset',
                              'experiment_idx',
                              'method',
                              'test_avg_IBS',
                              'test_IBS',
                              'test_cindex_td'] + \
                             [f'test_BS_q{int(q*100)}' for q in eval_horizon_quantiles] + \
                             [f'test_cr_cindex_q{int(q*100)}' for q in eval_horizon_quantiles])

for experiment_idx in range(n_experiment_repeats):
    for dataset in datasets:
        X_full_train_raw_np, Y_full_train_np, D_full_train_np, \
                X_test_raw_np, Y_test_np, D_test_np, \
                features_before_preprocessing, features_after_preprocessing, \
                events, train_test_split_prespecified, \
                build_preprocessor_and_preprocess, apply_preprocessor = load_dataset(dataset, experiment_idx, competing=True)
        
        # set the evaluation horizon based on the entire dataset to allow fair comparison
        Y_all, D_all = np.concatenate([Y_full_train_np, Y_test_np]), np.concatenate([D_full_train_np, D_test_np])
        eval_horizons = np.quantile(Y_all[D_all>0], eval_horizon_quantiles)
        # set the integrated Brier score horizons, which is used for model selection during hyperparameter search / early stopping
        # set the t_min to be the minimum observed event time 
        # avoid setting t_max to be the maximum observed event time due to instability
        tau = np.percentile(Y_all[D_all > 0], ibs_max_horizon_percentile*100)  # e.g. 90th percentile event time
        ibs_integrate_horizons = np.linspace(Y_all[D_all > 0].min(), tau, ibs_n_horizon_points)
        all_eval_horizons = np.sort(np.concatenate([eval_horizons, ibs_integrate_horizons]))
        
        # split the "full training set" into the actual training set and a validation set (using a 80/20 split)
        X_train_raw_np, X_val_raw_np, Y_train_np, Y_val_np, D_train_np, D_val_np = \
            train_test_split(X_full_train_raw_np, Y_full_train_np, D_full_train_np,
                            test_size=.2, random_state=0)
        

        print('[Dataset: %s, experiment: %d]' % (dataset, experiment_idx))
        print()

        print(f'Training set size {X_train_raw_np.shape[0]}')
        print(f'Validation set size {X_val_raw_np.shape[0]}')
        print(f'Test set size {X_test_raw_np.shape[0]}')
        print()

        print(f'Features before preprocessing ({len(features_before_preprocessing)} total):')
        print(features_before_preprocessing)
        print()

        print(f'Features after preprocessing ({len(features_after_preprocessing)} total):')
        print(features_after_preprocessing)
        print()

        print('Events:', events)
        print()

        # fit and apply a preprocessor to the training set; apply (but do not re-fit) the preprocessor to the validation set.
        X_train_np, preprocessor = build_preprocessor_and_preprocess(X_train_raw_np)
        X_val_np = apply_preprocessor(X_val_raw_np, preprocessor)

        X_train_np, X_val_np = X_train_np.astype('float32'), X_val_np.astype('float32')
        Y_train_np, Y_val_np = Y_train_np.astype('float32'), Y_val_np.astype('float32')
        D_train_np, D_val_np = D_train_np.astype('float32'), D_val_np.astype('float32')

        # fit KM estimator for censoring
        censoring_kmf = KaplanMeierFitter()
        censoring_kmf.fit(durations=Y_train_np, event_observed=1 * (D_train_np==0))

        output_train_metrics_filename \
            = os.path.join(output_dir, 'train',
                           '%s_%s_exp%d_%s_train_metrics_%s.txt'
                           % (estimator_name, dataset, experiment_idx,
                              validation_string, hyperparam_hash))
        output_best_hyperparam_filename \
            = os.path.join(output_dir, 'train',
                           '%s_%s_exp%d_%s_best_hyperparams_%s.pkl'
                           % (estimator_name, dataset, experiment_idx,
                              validation_string, hyperparam_hash))
        if not os.path.isfile(output_train_metrics_filename) or \
                not os.path.isfile(output_best_hyperparam_filename):
            print('Training...', flush=True)
            train_metrics_file = open(output_train_metrics_filename, 'w')
            best_hyperparams = {}

            min_loss = np.inf
            arg_min = None
            best_model_filename = 'cache_' + str(uuid.uuid4()) + '.pt'

            for hyperparam_idx, hyperparam in enumerate(hyperparams):
                alpha, sigma, n_durations, n_layers, n_nodes, batch_size, \
                    max_n_epochs, lr = hyperparam

                if alpha == 0 and sigma != sigma_range[0]:
                    continue

                # seed different hyperparameters differently to prevent weird
                # behavior where a bad initial seed makes a specific model
                # always look terrible
                hyperparam_random_seed = method_random_seed + hyperparam_idx

                dataset_max_n_epochs = 'max_n_epochs_%s' % dataset
                if dataset_max_n_epochs in config[method_header]:
                    max_n_epochs = \
                        int(config[method_header][dataset_max_n_epochs])

                tic = time.time()
                torch.manual_seed(hyperparam_random_seed)
                np.random.seed(hyperparam_random_seed)
                random.seed(hyperparam_random_seed)

                optimizer = tt.optim.Adam(lr=lr)
                if n_durations == 0:
                    # use all unique non-censored event times, with a upper limit of 512
                    mask = (D_train_np >= 1)  # boolean mask specifying which training points were not censored
                    n_unique_times = np.unique(Y_train_np[mask]).shape[0]
                    if n_unique_times > 512:
                        print(f'Trying to use all training unique event times, found {n_unique_times} unique event times, using upper limit of 512 instead.')
                        label_transform = LabTransformCR(512, scheme='quantiles')
                    else:
                        label_transform = LabTransformCR(np.unique(Y_train_np[mask]))
                else:
                    label_transform = LabTransformCR(n_durations, scheme='quantiles')
                Y_train_discrete_np, D_train_discrete_np = label_transform.fit_transform(Y_train_np, D_train_np)
                Y_val_discrete_np, D_val_discrete_np = label_transform.transform(Y_val_np, D_val_np)
                net = CauseSpecificNet(X_train_np.shape[1],
                                       [n_nodes for _ in range(n_layers)],
                                       [n_nodes for _ in range(n_layers)],
                                       len(events),
                                       label_transform.out_features,
                                       True,
                                       0.)
                model = DeepHit(net, optimizer, alpha=alpha, sigma=sigma,
                                duration_index=label_transform.cuts)
                
                time_elapsed_filename = \
                    os.path.join(output_dir, 'models',
                                 '%s_%s_exp%d_a%f_s%f_nd%d_nla%d_nno%d_'
                                 % (estimator_name, dataset, experiment_idx,
                                    alpha, sigma, n_durations, n_layers,
                                    n_nodes)
                                 +
                                 'bs%d_mnep%d_lr%f_time.txt'
                                 % (batch_size, max_n_epochs, lr))
                epoch_time_elapsed_filename = \
                    time_elapsed_filename[:-8] + 'epoch_times.txt'
                epoch_times = []

                train_pair = (X_train_np, (Y_train_discrete_np, D_train_discrete_np))
                model.training_data = tt.tuplefy(*train_pair)
                train_loader = \
                    model.make_dataloader(train_pair, batch_size, True)
                best_loss = np.inf
                for epoch_idx in range(max_n_epochs):
                    tic_ = time.time()
                    model.fit_dataloader(train_loader, epochs=1)
                    epoch_train_time = time.time() - tic_
                    tic_ = time.time()
                    surv_df = model.predict_surv_df(X_val_np)
                    y_val_pred = (surv_df.to_numpy(), surv_df.index)
                    cif_pred_all_events = model.predict_cif(X_val_np) # shape (n_events, n_durations, n_patients)
                    ibs_all_events, ctd_all_events = [], []
                    for e_idx_minus_1, event in enumerate(events):
                        cif_val_pred_event = cif_pred_all_events[e_idx_minus_1]  # shape (n_durations, n_patients)
                        # Interpolation: evaluate CIF for each patient at all_eval_horizons
                        cif_interp = np.empty((len(all_eval_horizons), cif_val_pred_event.shape[1]))
                        for j in range(cif_val_pred_event.shape[1]):
                            cif_interp[:, j] = np.interp(
                                all_eval_horizons, 
                                model.duration_index, 
                                cif_val_pred_event[:, j]
                                )
                        # Wrap into a DataFrame for convenience
                        cif_interp_df = pd.DataFrame(
                            cif_interp,
                            index=all_eval_horizons,
                            columns=[f"patient_{j}" for j in range(cif_val_pred_event.shape[1])]
                        )
                        cif_eval_df = cif_interp_df.loc[all_eval_horizons[np.isin(all_eval_horizons, eval_horizons)]] # select the entries for evaluation
                        cif_ibs_df  = cif_interp_df.loc[all_eval_horizons[np.isin(all_eval_horizons, ibs_integrate_horizons)]] # select the entries for IBS integration
                        ibs = compute_ibs_competing(
                            cif_ibs_df.values.T, censoring_kmf,
                            Y_val_np, D_val_np, e_idx_minus_1+1, cif_ibs_df.index)
                        cindex_td = -neg_cindex_td(Y_val_np, (D_val_np==e_idx_minus_1+1).astype(int), 
                                                   (-cif_ibs_df.values, cif_ibs_df.index), exact=False)
                        ibs_all_events.append(ibs)
                        ctd_all_events.append(cindex_td)

                    avg_ibs = np.array(ibs_all_events).mean()
                    avg_ctd = np.array(ctd_all_events).mean()
                    val_loss = avg_ibs if model_selection_metric == 'avgIBS' else -avg_ctd
                    epoch_val_time = time.time() - tic_
                    epoch_times.append([epoch_train_time,
                                        epoch_val_time])
                    if val_loss != 0.:
                        new_hyperparam = (alpha, sigma, n_durations,
                                          n_layers, n_nodes, batch_size,
                                          epoch_idx + 1, lr,
                                          hyperparam_random_seed)
                        print(new_hyperparam,
                              '--',
                              'val avg IBS %.4f' % avg_ibs,
                              '--',
                              'val avg Ctd %.4f' % avg_ctd,
                              '--',
                              'train time %f sec(s)'
                              % epoch_train_time,
                              '--',
                              'val time %f sec(s)'
                              % epoch_val_time,
                              flush=True)
                        print(new_hyperparam, ':', val_loss, flush=True,
                              file=train_metrics_file)

                        if val_loss < best_loss:
                            best_loss = val_loss
                            wait_idx = 0

                            if val_loss < min_loss:
                                min_loss = val_loss
                                arg_min = new_hyperparam
                                model.save_net(best_model_filename)
                        else:
                            wait_idx += 1
                            if patience > 0 and wait_idx >= patience:
                                break
                    else:
                        # weird corner case
                        break

                np.savetxt(epoch_time_elapsed_filename,
                           np.array(epoch_times))

                elapsed = time.time() - tic
                print('Time elapsed: %f second(s)' % elapsed, flush=True)
                np.savetxt(time_elapsed_filename,
                           np.array(elapsed).reshape(1, -1))

                del model
                gc.collect()
            
            train_metrics_file.close()

            best_hyperparams['loss'] = (arg_min, min_loss)

            alpha, sigma, n_durations, n_layers, n_nodes, batch_size, \
                n_epochs, lr, seed = arg_min
            model_filename = \
                os.path.join(output_dir, 'models',
                             '%s_%s_exp%d_a%f_s%f_nd%d_nla%d_nno%d_'
                             % (estimator_name, dataset, experiment_idx,
                                alpha, sigma, n_durations, n_layers,
                                n_nodes)
                             +
                             'bs%d_nep%d_lr%f_test.pt'
                             % (batch_size, n_epochs, lr))
            time_elapsed_filename = \
                os.path.join(output_dir, 'models',
                             '%s_%s_exp%d_a%f_s%f_nd%d_nla%d_nno%d_'
                             % (estimator_name, dataset, experiment_idx,
                                alpha, sigma, n_durations, n_layers,
                                n_nodes)
                             +
                             'bs%d_mnep%d_lr%f_time.txt'
                             % (batch_size, max_n_epochs, lr))
            os.rename(best_model_filename, model_filename)
            shutil.copy(time_elapsed_filename,
                        model_filename[:-3] + '_time.txt')

            with open(output_best_hyperparam_filename, 'wb') as pickle_file:
                pickle.dump(best_hyperparams, pickle_file,
                            protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print('Loading previous validation results...', flush=True)
            with open(output_best_hyperparam_filename, 'rb') as pickle_file:
                best_hyperparams = pickle.load(pickle_file)
            arg_min, min_loss = best_hyperparams['loss']

        print('Best hyperparameters for minimizing loss:',
              arg_min, '-- achieves val avg IBS %f' % min_loss, flush=True)
        
        print()
        print('Testing...', flush=True)
        X_test_np = apply_preprocessor(X_test_raw_np, preprocessor)
        X_test_np = X_test_np.astype('float32')
        Y_test_np = Y_test_np.astype('float32')
        D_test_np = D_test_np.astype('float32')

        alpha, sigma, n_durations, n_layers, n_nodes, batch_size, n_epochs, \
            lr, seed = arg_min
        
        tic = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        optimizer = tt.optim.Adam(lr=lr)
        if n_durations == 0:
            # use all unique non-censored event times, with a upper limit of 512
            mask = (D_train_np >= 1)  # boolean mask specifying which training points were not censored
            n_unique_times = np.unique(Y_train_np[mask]).shape[0]
            if n_unique_times > 512:
                print(f'Trying to use all training unique event times, found {n_unique_times} unique event times, using upper limit of 512 instead.')
                label_transform = LabTransformCR(512, scheme='quantiles')
            else:
                label_transform = LabTransformCR(np.unique(Y_train_np[mask]))
        else:
            label_transform = LabTransformCR(n_durations, scheme='quantiles')
        Y_train_discrete_np, D_train_discrete_np = label_transform.fit_transform(Y_train_np, D_train_np)

        net = CauseSpecificNet(X_train_np.shape[1],
                               [n_nodes for _ in range(n_layers)],
                               [n_nodes for _ in range(n_layers)],
                               len(events),
                               label_transform.out_features,
                               True,
                               0.)
        model = DeepHit(net, optimizer, alpha=alpha, sigma=sigma,
                        duration_index=label_transform.cuts)
        
        model_filename = \
            os.path.join(output_dir, 'models',
                         '%s_%s_exp%d_a%f_s%f_nd%d_nla%d_nno%d_'
                         % (estimator_name, dataset, experiment_idx,
                            alpha, sigma, n_durations, n_layers, n_nodes)
                         +
                         'bs%d_nep%d_lr%f_test.pt'
                         % (batch_size, n_epochs, lr))
        time_elapsed_filename = model_filename[:-3] + '_time.txt'
        model.load_net(model_filename)
        elapsed = float(np.loadtxt(time_elapsed_filename))
        print('Time elapsed (from previous fitting): %f second(s)'
              % elapsed, flush=True)
        
        surv_df = model.predict_surv_df(X_test_np)
        y_test_pred = (surv_df.to_numpy(), surv_df.index)
        cif_test_pred_all_events = model.predict_cif(X_test_np) # shape (n_events, n_durations, n_patients)
        test_eval_brier_scores_all_events, test_ibs_all_events = [], []
        test_cindex_td_all_events, test_eval_cindex_cr_all_events = [], []
        for e_idx_minus_1, event in enumerate(events):
            cif_test_pred_event = cif_test_pred_all_events[e_idx_minus_1] # shape (n_durations, n_patients)
            # Interpolation: evaluate CIF for each patient at all_eval_horizons with linear interpolation
            cif_interp = np.empty((len(all_eval_horizons), cif_test_pred_event.shape[1]))
            for j in range(cif_test_pred_event.shape[1]):  # loop over patients
                cif_interp[:, j] = np.interp(
                    all_eval_horizons,           # new evaluation times
                    model.duration_index,        # original grid
                    cif_test_pred_event[:, j]    # CIF trajectory for patient j
                    )
            cif_interp_df = pd.DataFrame(
                cif_interp,
                index=all_eval_horizons,
                columns=[f"patient_{j}" for j in range(cif_test_pred_event.shape[1])]
            )
            cif_eval_df = cif_interp_df.loc[all_eval_horizons[np.isin(all_eval_horizons, eval_horizons)]] # select the entries for evaluation
            cif_ibs_df  = cif_interp_df.loc[all_eval_horizons[np.isin(all_eval_horizons, ibs_integrate_horizons)]] # select the entries for IBS integration
            eval_brier_scores = compute_brier_competing_multiple_times(
                cif_eval_df.values.T, censoring_kmf,
                Y_test_np, D_test_np, e_idx_minus_1+1, cif_eval_df.index)
            ibs = compute_ibs_competing(
                cif_ibs_df.values.T, censoring_kmf,
                Y_test_np, D_test_np, e_idx_minus_1+1, cif_ibs_df.index)
            cindex_td = -neg_cindex_td(Y_test_np, (D_test_np==e_idx_minus_1+1).astype(int), 
                                       (-cif_ibs_df.values, cif_ibs_df.index), exact=False)
            eval_cindex_cr = [0
                # c_index_competing_single_time(
                # Y_test_np, cif_eval_df.values[h, :], 
                # D_test_np, e_idx_minus_1+1) 
                for h in range(len(eval_horizons))]
            test_eval_brier_scores_all_events.append(eval_brier_scores)
            test_ibs_all_events.append(ibs)
            test_cindex_td_all_events.append(cindex_td)
            test_eval_cindex_cr_all_events.append(eval_cindex_cr)

        test_avg_IBS = np.array(test_ibs_all_events).mean() # average IBS across events with equal weights
        test_eval_brier_scores_all_events = np.array(test_eval_brier_scores_all_events) # shape (n_events, n_eval_horizon_quantiles)
        test_eval_cindex_cr_all_events = np.array(test_eval_cindex_cr_all_events) # shape (n_events, n_eval_horizon_quantiles)
        print('Hyperparameter', arg_min, 'achieves test avg IBS: %f' % test_avg_IBS,
              flush=True)

        final_test_scores = {}
        test_set_metrics = [test_avg_IBS]
        test_set_metrics.extend([test_ibs_all_events])
        test_set_metrics.extend([test_cindex_td_all_events])
        test_set_metrics.extend([[e for e in test_eval_brier_scores_all_events[:, h]] for h in range(len(eval_horizon_quantiles))])
        test_set_metrics.extend([[e for e in test_eval_cindex_cr_all_events[:, h]] for h in range(len(eval_horizon_quantiles))])
        final_test_scores[arg_min] = tuple(test_set_metrics)

        del model
        gc.collect()

        test_csv_writer.writerow(
            [dataset, experiment_idx, estimator_name] + test_set_metrics)
        output_test_table_file.flush()

        print()
        print()

output_test_table_file.close()