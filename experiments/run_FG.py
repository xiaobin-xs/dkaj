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

from datasets import load_dataset
from modelsR import FineGrayCRR

from metrics import neg_cindex_td, c_index_competing_single_time, compute_brier_competing_multiple_times, compute_ibs_competing



estimator_name = 'fine_gray'

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

hyperparams = [''] # no hyperparameter search for Fine-Gray model

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
# if output_test_table_filename exist:
# open it with append mode
if os.path.isfile(output_test_table_filename):
    print('Appending to existing test results file %s'
          % output_test_table_filename)
    output_test_table_file = open(output_test_table_filename, 'a')
    test_csv_writer = csv.writer(output_test_table_file)
else:
    print('Creating new test results file %s'
          % output_test_table_filename)
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
        X_train_np, preprocessor = build_preprocessor_and_preprocess(X_train_raw_np, cox=True)
        X_val_np = apply_preprocessor(X_val_raw_np, preprocessor, cox=True)

        # fit KM estimator for censoring
        censoring_kmf = KaplanMeierFitter()
        censoring_kmf.fit(durations=Y_train_np, event_observed=1 * (D_train_np==0))

        output_train_metrics_filename \
            = os.path.join(output_dir, 'train',
                           '%s_%s_exp%d_%s_train_metrics_%s.txt'
                           % (estimator_name, dataset, experiment_idx,
                              validation_string, hyperparam_hash))
        print('Training...', flush=True)
        train_metrics_file = open(output_train_metrics_filename, 'w')

        tic = time.time()

        model = FineGrayCRR(n_causes=len(events), interpolate='linear')
        time_elapsed_filename = \
            os.path.join(output_dir, 'models',
                            '%s_%s_exp%d_'
                            % (estimator_name, dataset, experiment_idx)
                            + 'time.txt')
        epoch_time_elapsed_filename = \
            time_elapsed_filename[:-8] + 'epoch_times.txt'

        tic_ = time.time()
        # No hyperparam search, fit with the entire training set
        X_train_val_np = np.concatenate([X_train_np, X_val_np], axis=0)
        Y_train_val_np = np.concatenate([Y_train_np, Y_val_np], axis=0)
        D_train_val_np = np.concatenate([D_train_np, D_val_np], axis=0)
        model.fit(X_train_val_np, Y_train_val_np, D_train_val_np)
        avg_epoch_train_time = time.time() - tic_

        tic_ = time.time()
        cif_pred_all_events = model.predict_cumulative_incidence(X_val_np, times=all_eval_horizons) # (n_samples, n_events, n_times)
        ibs_all_events, ctd_all_events = [], []
        for e_idx_minus_1, event in enumerate(events):
            cif_val_pred_event = cif_pred_all_events[:, e_idx_minus_1, :] # (n_samples, n_times)
            cif_df = pd.DataFrame(
                cif_val_pred_event.T,
                index=all_eval_horizons,
                columns=[f"patient_{j}" for j in range(cif_val_pred_event.shape[0])]
            )
            cif_eval_df = cif_df.loc[all_eval_horizons[np.isin(all_eval_horizons, eval_horizons)]] # select the entries for evaluation
            cif_ibs_df = cif_df.loc[all_eval_horizons[np.isin(all_eval_horizons, ibs_integrate_horizons)]] # select the entries for IBS integration
            ibs = compute_ibs_competing(
                cif_ibs_df.values.T, censoring_kmf,
                Y_val_np, D_val_np, e_idx_minus_1+1, cif_ibs_df.index)
            cindex_td = -neg_cindex_td(Y_val_np, (D_val_np==e_idx_minus_1+1).astype(int), 
                                            (-cif_ibs_df.values, cif_ibs_df.index), exact=False)
            ibs_all_events.append(ibs)
            ctd_all_events.append(cindex_td)
        epoch_val_time = time.time() - tic_

        epoch_times = [avg_epoch_train_time, epoch_val_time]

        avg_ibs = np.array(ibs_all_events).mean()
        avg_ctd = np.array(ctd_all_events).mean()
        val_loss = avg_ibs if model_selection_metric == 'avgIBS' else -avg_ctd
        
        print(
                'val avg IBS %.4f' % avg_ibs,
                '; ',
                'val avg Ctd %.4f' % avg_ctd,
                flush=True)

        np.savetxt(epoch_time_elapsed_filename,
                    np.array(epoch_times))
        
        elapsed = time.time() - tic
        print('Time elapsed: %f second(s)' % elapsed, flush=True)
        np.savetxt(time_elapsed_filename,
                    np.array(elapsed).reshape(1, -1))
        
        train_metrics_file.close()

        model_filename = \
            os.path.join(output_dir, 'models',
                            '%s_%s_exp%d_'
                            % (estimator_name, dataset, experiment_idx)
                            + 'test.pkl')
        time_elapsed_filename = \
            os.path.join(output_dir, 'models',
                            '%s_%s_exp%d_'
                            % (estimator_name, dataset, experiment_idx)
                            + 'time.txt')
        shutil.copy(time_elapsed_filename,
                    model_filename[:-4] + '_time.txt')

        print()
        print('Testing...', flush=True)
        X_test_np = apply_preprocessor(X_test_raw_np, preprocessor, cox=True)
        
        tic = time.time()
        
        time_elapsed_filename = model_filename[:-4] + '_time.txt'
        elapsed = float(np.loadtxt(time_elapsed_filename))
        print('Time elapsed (from previous fitting): %f second(s)'
              % elapsed, flush=True)

        cif_pred_all_events = model.predict_cumulative_incidence(X_test_np, times=all_eval_horizons) # (n_samples, n_events + 1, n_times)
        test_eval_brier_scores_all_events, test_ibs_all_events = [], []
        test_cindex_td_all_events, test_eval_cindex_cr_all_events = [], []
        for e_idx_minus_1, event in enumerate(events):
            cif_test_pred_event = cif_pred_all_events[:, e_idx_minus_1, :] # (n_samples, n_times)
            cif_df = pd.DataFrame(
                        cif_test_pred_event.T,
                        index=all_eval_horizons,
                        columns=[f"patient_{j}" for j in range(cif_test_pred_event.shape[0])]
                    )
            cif_eval_df = cif_df.loc[all_eval_horizons[np.isin(all_eval_horizons, eval_horizons)]] # select the entries for evaluation
            cif_ibs_df  = cif_df.loc[all_eval_horizons[np.isin(all_eval_horizons, ibs_integrate_horizons)]] # select the entries for IBS integration
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
        print('Fine and Gray achieves test avg IBS: %f' % test_avg_IBS,
              flush=True)

        # final_test_scores = {}
        test_set_metrics = [test_avg_IBS]
        test_set_metrics.extend([test_ibs_all_events])
        test_set_metrics.extend([test_cindex_td_all_events])
        test_set_metrics.extend([[e for e in test_eval_brier_scores_all_events[:, h]] for h in range(len(eval_horizon_quantiles))])
        test_set_metrics.extend([[e for e in test_eval_cindex_cr_all_events[:, h]] for h in range(len(eval_horizon_quantiles))])
        # final_test_scores[arg_min] = tuple(test_set_metrics)

        del model
        gc.collect()

        test_csv_writer.writerow(
            [dataset, experiment_idx, estimator_name] + test_set_metrics)
        output_test_table_file.flush()

        print()
        print()

output_test_table_file.close()