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
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from lifelines import KaplanMeierFitter
from copy import deepcopy

import torchtuples as tt

from datasets import load_dataset, LabTransformCR
from models import SurvivalBoostWrap, hist_gradient_boosting_classifier_apply
from models import create_base_neural_net_with_hypersphere, tuna_loss
from models import DKAJ, DKAJSummaryLoss, DKAJSummary
from metrics import neg_cindex_td, c_index_competing_single_time, compute_brier_competing_multiple_times, compute_ibs_competing

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
estimator_name = 'survboost'
finetune_estimator_name = 'dkaj_noTuna'

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
output_dir = config['DEFAULT']['init_output_dir']
finetune_output_dir = config['DEFAULT']['output_dir']
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

n_neighbors_range = ast.literal_eval(config['DEFAULT']['ANN_max_n_neighbors'])

compute_bootstrap_CI = int(config['DEFAULT']['compute_bootstrap_CI']) > 0
bootstrap_CI_coverage = float(config['DEFAULT']['bootstrap_CI_coverage'])
bootstrap_n_samples = int(config['DEFAULT']['bootstrap_n_samples'])
bootstrap_random_seed = int(config['DEFAULT']['bootstrap_random_seed'])

tuna_random_seed = int(config['DEFAULT']['tuna_random_seed'])

if model_selection_metric == 'avgCtd':
    output_dir += f'_{model_selection_metric}'
    finetune_output_dir += f'_{model_selection_metric}'
os.makedirs(finetune_output_dir, exist_ok=True)
os.makedirs(os.path.join(finetune_output_dir, 'models'), exist_ok=True)
os.makedirs(os.path.join(finetune_output_dir, 'train'), exist_ok=True)

# hyperparams = \
#     [(lr, n_iter, max_depth, n_time_grid_steps, ipcw_strategy)
#      for lr in ast.literal_eval(config[method_header]['learning_rate'])
#          for n_iter in ast.literal_eval(config[method_header]['n_iter'])
#          for max_depth in ast.literal_eval(config[method_header]['max_depth'])
#          for n_time_grid_steps in ast.literal_eval(config[method_header]['n_time_grid_steps'])
#          for ipcw_strategy in ast.literal_eval(config[method_header]['ipcw_strategy'])
#     ]

finetune_method_header = 'method: %s' % finetune_estimator_name.split('_')[0]
finetune_method_random_seed = \
    int(config[finetune_method_header]['random_seed'])
finetune_summaries = \
    int(config[finetune_method_header]['finetune_summaries'])
max_n_epochs = int(config[finetune_method_header]['max_n_epochs'])
lr_range = ast.literal_eval(config[finetune_method_header]['learning_rate'])
# if 'warmstart_learning_rate' in config[finetune_method_header]:
#     warmstart_lr_range = \
#         ast.literal_eval(
#             config[finetune_method_header]['warmstart_learning_rate'])
# else:
#     warmstart_lr_range = lr_range
# warmstart_hyperparams = \
#     [(squared_radius, n_layers, n_nodes, batch_size, max_n_epochs, lr)
#      for squared_radius
#      in ast.literal_eval(config[finetune_method_header]['squared_radius'])
#      for n_layers
#      in ast.literal_eval(config[finetune_method_header]['n_layers'])
#      for n_nodes
#      in ast.literal_eval(config[finetune_method_header]['n_nodes'])
#      for batch_size
#      in ast.literal_eval(config[finetune_method_header]['batch_size'])
#      for lr
#      in warmstart_lr_range]
sigma_range = ast.literal_eval(config[finetune_method_header]['sigma'])
n_durations_range = \
    ast.literal_eval(config[finetune_method_header]['n_durations'])
gamma_range = ast.literal_eval(config[finetune_method_header]['gamma'])
finetune_hyperparams = \
    [(squared_radius, n_layers, n_nodes, batch_size, alpha, sigma, n_durations, gamma, beta, min_kernel_weight,
      n_neighbors, batch_size, max_n_epochs, lr)
     for squared_radius
     in ast.literal_eval(config[finetune_method_header]['squared_radius'])
     for n_layers
     in ast.literal_eval(config[finetune_method_header]['n_layers'])
     for n_nodes
     in ast.literal_eval(config[finetune_method_header]['n_nodes'])
     for batch_size
     in ast.literal_eval(config[finetune_method_header]['batch_size'])
     for alpha
     in ast.literal_eval(config[finetune_method_header]['alpha'])
     for sigma
     in sigma_range
     for n_durations
     in n_durations_range
     for gamma
     in gamma_range
     for beta
     in ast.literal_eval(config[finetune_method_header]['beta'])
     for min_kernel_weight
     in ast.literal_eval(config[finetune_method_header]['min_kernel_weight'])
     for n_neighbors
     in n_neighbors_range
     for batch_size
     in ast.literal_eval(config[finetune_method_header]['batch_size'])
     for lr
     in lr_range]
if 'sumtune_learning_rate' in config[finetune_method_header]:
    sumtune_lr_range = \
        ast.literal_eval(
            config[finetune_method_header]['sumtune_learning_rate'])
else:
    sumtune_lr_range = lr_range

# hyperparam_hash = hashlib.sha256()
# hyperparam_hash.update(str(hyperparams).encode('utf-8'))
# hyperparam_hash = hyperparam_hash.hexdigest()

warmstart_hyperparam_hash = hashlib.sha256()
# warmstart_hyperparam_hash.update(str(hyperparams).encode('utf-8'))
# warmstart_hyperparam_hash.update(str(warmstart_hyperparams).encode('utf-8'))
warmstart_hyperparam_hash = warmstart_hyperparam_hash.hexdigest()

finetune_hyperparam_hash = hashlib.sha256()
# finetune_hyperparam_hash.update(str(hyperparams).encode('utf-8'))
# finetune_hyperparam_hash.update(str(warmstart_hyperparams).encode('utf-8'))
finetune_hyperparam_hash.update(str(finetune_hyperparams).encode('utf-8'))
finetune_hyperparam_hash = finetune_hyperparam_hash.hexdigest()

validation_string = 'vr%f' % val_ratio

full_estimator_name = finetune_estimator_name # '%s_tuna=%s' % (finetune_estimator_name, estimator_name)

output_test_table_filename \
    = os.path.join(finetune_output_dir,
                   '%s_experiments%d_%s_test_metrics_%s%d.csv'
                   % (full_estimator_name,
                      n_experiment_repeats,
                      validation_string,
                      finetune_hyperparam_hash,
                      finetune_summaries))
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

n_random_times_per_data_point = int(config[finetune_method_header]['n_random_times_per_data_point'])

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

        # output_train_metrics_filename \
        #     = os.path.join(output_dir, 'train',
        #                    '%s_%s_exp%d_%s_train_metrics_%s.txt'
        #                    % (estimator_name, dataset, experiment_idx,
        #                       validation_string, hyperparam_hash))
        # output_best_hyperparam_filename \
        #     = os.path.join(output_dir, 'train',
        #                    '%s_%s_exp%d_%s_best_hyperparams_%s.pkl'
        #                    % (estimator_name, dataset, experiment_idx,
        #                       validation_string, hyperparam_hash))
        # if not os.path.isfile(output_train_metrics_filename):
        #     raise Exception('File not found: '
        #                     + output_train_metrics_filename)
        # if not os.path.isfile(output_best_hyperparam_filename):
        #     raise Exception('File not found: '
        #                     + output_best_hyperparam_filename)
        
        # with open(output_best_hyperparam_filename, 'rb') as pickle_file:
        #     best_hyperparams = pickle.load(pickle_file)
        # best_hyperparam, min_loss = best_hyperparams['loss']

        # lr, n_iter, max_depth, n_time_grid_steps, ipcw_strategy, \
        #     survboost_seed = best_hyperparam
        
        # model_filename = \
        #         os.path.join(output_dir, 'models',
        #                      '%s_%s_exp%d_dpt%s_nd%d_%s_'
        #                      % (estimator_name, dataset, experiment_idx,
        #                         max_depth, n_time_grid_steps, ipcw_strategy)
        #                      + 'iter%d_lr%f_test.pkl'
        #                      % (n_iter, lr))
        # if not os.path.isfile(model_filename):
        #     raise Exception('File not found: ' + model_filename)
        
        # survival_boost = SurvivalBoostWrap.load_model(model_filename)
        # np.random.seed(survboost_seed)
        
        # X_train_with_random_times_np = []
        # X_val_with_random_times_np = []
        # select random times for each data point for survival boost model inference
        # for idx in range(n_random_times_per_data_point):
        #     random_times = np.random.uniform(Y_train_np.min(), Y_train_np.max(),
        #                                     size=X_train_np.shape[0]).reshape(-1, 1) # (n_samples, 1)
        #     X_train_with_random_times_np.append(np.hstack([random_times, X_train_np]))
        #     random_times = np.random.uniform(Y_val_np.min(), Y_val_np.max(),
        #                                     size=X_val_np.shape[0]).reshape(-1, 1)
        #     X_val_with_random_times_np.append(np.hstack([random_times, X_val_np]))
        # X_train_with_random_times_np = np.vstack(X_train_with_random_times_np)       # (n_samples*n_random_times, 1+n_features)
        # X_val_with_random_times_np = np.vstack(X_val_with_random_times_np)
        # predict the leaves
        # survival_boost_leaves_train_np = \
        #     hist_gradient_boosting_classifier_apply(survival_boost.estimator_,
        #                                             X_train_with_random_times_np,
        #                                             os.cpu_count()).reshape(X_train_np.shape[0], -1) # before reshape: (n_samples*n_random_times, n_iter_gb, n_class=1+n_events)
        # survival_boost_leaves_val_np = \
        #     hist_gradient_boosting_classifier_apply(survival_boost.estimator_,
        #                                             X_val_with_random_times_np,
        #                                             os.cpu_count()).reshape(X_val_np.shape[0], -1)
        # prepare data for TUNA warming
        # X_train = torch.tensor(X_train_np, dtype=torch.float32, device=device)
        # X_train_repeat = X_train.repeat(n_random_times_per_data_point, 1)
        # leaves_train = torch.tensor(survival_boost_leaves_train_np, dtype=torch.float32, device=device)
        # tuna_train_data = list(zip(X_train_repeat, leaves_train))

        # X_val = torch.tensor(X_val_np, dtype=torch.float32, device=device)
        # X_val_repeat = X_val.repeat(n_random_times_per_data_point, 1)
        # leaves_val = torch.tensor(survival_boost_leaves_val_np, dtype=torch.float32, device=device)
        # tuna_val_data = list(zip(X_val_repeat, leaves_val))

        # ---------------------------------------------------------------------

        # warmstart_best_params_filename = \
        #     model_filename[:-6] \
        #     + '_tuna_mlp_params_%s.pkl' \
        #     % (warmstart_hyperparam_hash)
        # warmstart_train_times_filename = \
        #     model_filename[:-6] \
        #     + '_tuna_training_times_%s.txt' \
        #     % (warmstart_hyperparam_hash)
        # if not os.path.isfile(warmstart_best_params_filename) \
        #         or not os.path.isfile(warmstart_train_times_filename):
        #     warmstart_min_score = np.inf
        #     warmstart_best_params = None
        #     warmstart_train_times = []

        #     min_loss = np.inf
        #     arg_min = None

        #     cache_warmstart_model_filename = 'cache_' + str(uuid.uuid4()) + '.pt'
            
        #     for warmstart_hyperparam in warmstart_hyperparams:
        #         squared_radius, n_layers, n_nodes, batch_size, max_n_epochs, \
        #             lr = warmstart_hyperparam

        #         n_output_features = min(n_nodes, X_train_np.shape[1])

        #         dataset_max_n_epochs = 'max_n_epochs_%s' % dataset
        #         if dataset_max_n_epochs in config[finetune_method_header]:
        #             max_n_epochs = \
        #                 int(config[finetune_method_header][
        #                     dataset_max_n_epochs])

        #         print()

        #         tic = time.time()

        #         torch.manual_seed(tuna_random_seed)
        #         np.random.seed(tuna_random_seed)
        #         random.seed(tuna_random_seed)

        #         num_input_features = X_train_np.shape[1]
        #         tuna_train_loader = DataLoader(tuna_train_data, batch_size, shuffle=True)  # shuffling for minibatch gradient descent
        #         tuna_val_loader = DataLoader(tuna_val_data, batch_size, shuffle=False)  # there is no need to shuffle the validation data

        #         if squared_radius == 0:
        #             raise ValueError('squared radius equal to 0 not supported')
        #         else:
        #             base_neural_net = create_base_neural_net_with_hypersphere(
        #                 num_input_features,
        #                 [n_nodes for _ in range(n_layers)], squared_radius=squared_radius
        #                 ).to(device)
        #         optimizer = torch.optim.Adam(base_neural_net.parameters(), lr=lr)

        #         warmstart_model_filename = \
        #             model_filename[:-6] \
        #             + '_tuna_mlp_sr%f_nla%d_nno%d_bs%d_mnep%d_lr%f.pt' \
        #             % (squared_radius, n_layers, n_nodes, batch_size,
        #                max_n_epochs, lr)
        #         tuna_best_hyperparam_filename = \
        #             warmstart_model_filename[:-3] + '_best_hyperparams_%s.pkl' % (warmstart_hyperparam_hash)
        #         time_elapsed_filename = \
        #             warmstart_model_filename[:-3] + '_time.txt'
        #         if not os.path.isfile(warmstart_model_filename) or \
        #                 not os.path.isfile(time_elapsed_filename):
        #             print('Training warm-start neural net...')
        #             best_loss = np.inf
        #             for epoch_index in range(max_n_epochs):
        #                 base_neural_net.train()
        #                 for X_batch, leaves_batch in tuna_train_loader:
        #                     neural_net_output = base_neural_net(X_batch)
        #                     loss_batch = tuna_loss(neural_net_output, leaves_batch)

        #                     optimizer.zero_grad()
        #                     loss_batch.backward()
        #                     optimizer.step()

        #                 # TODO: change this to iBS?
        #                 # evaluate training and validation set losses
        #                 # (note that in practice, instead of evaluating the negative log likelihood loss,
        #                 # we could instead evaluate other metrics such as time-dependent concordance index,
        #                 # integrated Brier score, etc)
        #                 base_neural_net.eval()
        #                 with torch.no_grad():
        #                     train_loss = torch.tensor(0.0, dtype=torch.float, device=device)
        #                     num_points = 0
        #                     for X_batch, leaves_batch in tuna_train_loader:
        #                         batch_num_points = X_batch.size(0)
        #                         neural_net_output = base_neural_net(X_batch)
        #                         train_loss += tuna_loss(neural_net_output, leaves_batch) * batch_num_points
        #                         num_points += batch_num_points
        #                     train_loss = float(train_loss / num_points)
        #                     # train_epoch_losses.append(train_loss)
        #                     print(f'Epoch {epoch_index + 1} - train loss {train_loss}', end=' ', flush=True)

        #                     val_loss = torch.tensor(0.0, dtype=torch.float, device=device)
        #                     num_points = 0
        #                     for X_batch, leaves_batch in tuna_val_loader:
        #                         batch_num_points = X_batch.size(0)
        #                         neural_net_output = base_neural_net(X_batch)
        #                         val_loss += tuna_loss(neural_net_output, leaves_batch) * batch_num_points
        #                         num_points += batch_num_points
        #                     val_loss = float(val_loss / num_points)
        #                     # val_epoch_losses.append(val_loss)
        #                     print(f'- val loss {val_loss}', flush=True)

        #                 new_hyperparam = (squared_radius, n_layers, n_nodes, 
        #                                   batch_size, epoch_index+1, lr,
        #                                   tuna_random_seed)
        #                 if val_loss < best_loss:
        #                     best_loss = val_loss
        #                     # torch.save(base_neural_net.state_dict(), cache_warmstart_model_filename)
        #                     torch.save({
        #                         'model_state_dict': base_neural_net.state_dict(),
        #                         'val_loss': val_loss,
        #                         'new_hyperparam': new_hyperparam
        #                         }, cache_warmstart_model_filename)
        #                     wait_idx = 0

        #                 else:
        #                     wait_idx += 1
        #                     if patience > 0 and wait_idx >= patience:
        #                         break

        #             elapsed = time.time() - tic
        #             print('Time elapsed: %f second(s)' % elapsed, flush=True)
        #             os.rename(cache_warmstart_model_filename, warmstart_model_filename)
        #             np.savetxt(time_elapsed_filename,
        #                        np.array(elapsed).reshape(1, -1))
        #             # with open(tuna_best_hyperparam_filename, 'wb') as pickle_file:
        #             #     pickle.dump(best_hyperparams, pickle_file,
        #             #                 protocol=pickle.HIGHEST_PROTOCOL)
        #             warmstart_train_times.append(elapsed)
        #             warmstart_checkpoint = torch.load(warmstart_model_filename)
        #             score = warmstart_checkpoint['val_loss']
        #             print(warmstart_hyperparam, ':', score, flush=True)
        #             if score < warmstart_min_score:
        #                 warmstart_min_score = score
        #                 warmstart_best_params = \
        #                     (squared_radius, n_layers, n_nodes, batch_size,
        #                      max_n_epochs, lr)
        #         else:
        #             print('Loading previously fitted results...')
        #             warmstart_checkpoint = torch.load(warmstart_model_filename)
        #             base_neural_net.load_state_dict(warmstart_checkpoint['model_state_dict'])
        #             elapsed = float(np.loadtxt(time_elapsed_filename))
        #             print('Time elapsed (from previous fitting): %f second(s)'
        #                   % elapsed, flush=True)
        #             warmstart_train_times.append(elapsed)
        #             score = warmstart_checkpoint['val_loss']
        #             print(warmstart_hyperparam, ':', score, flush=True)
        #             if score < warmstart_min_score:
        #                 warmstart_min_score = score
        #                 warmstart_best_params = \
        #                     (squared_radius, n_layers, n_nodes, batch_size,
        #                      max_n_epochs, lr)
        #         print()

        #     with open(warmstart_best_params_filename, 'wb') as pickle_file:
        #         pickle.dump(warmstart_best_params, pickle_file,
        #                     protocol=pickle.HIGHEST_PROTOCOL)
        #     np.savetxt(warmstart_train_times_filename,
        #                np.array(warmstart_train_times))

        # print('Loading warm-start neural net...')
        # with open(warmstart_best_params_filename, 'rb') as pickle_file:
        #     warmstart_best_params = pickle.load(pickle_file)
        #     squared_radius, n_layers, n_nodes, batch_size, max_n_epochs, lr \
        #         = warmstart_best_params
        # torch.manual_seed(tuna_random_seed)
        # np.random.seed(tuna_random_seed)
        # random.seed(tuna_random_seed)

        num_input_features = X_train_np.shape[1]

        # if squared_radius == 0:
        #     raise ValueError('squared radius equal to 0 not supported')
        # else:
        #     base_neural_net = create_base_neural_net_with_hypersphere(
        #                 num_input_features,
        #                 [n_nodes for _ in range(n_layers)], squared_radius=squared_radius
        #                 ).to(device)

        # warmstart_model_filename = \
        #     model_filename[:-6] \
        #     + '_tuna_mlp_sr%f_nla%d_nno%d_bs%d_mnep%d_lr%f.pt' \
        #     % (squared_radius, n_layers, n_nodes, batch_size, max_n_epochs, lr)
        # warmstart_checkpoint = torch.load(warmstart_model_filename)
        # base_neural_net.load_state_dict(warmstart_checkpoint['model_state_dict'])
        # optimizer = torch.optim.Adam(base_neural_net.parameters(), lr=lr)

        # warmstart_min_score = warmstart_checkpoint['val_loss']
        # warmstart_train_times = \
        #     np.loadtxt(warmstart_train_times_filename)
        # print('Average validation batch kernel MSE:', warmstart_min_score)
        # print('Train times mean (std): %f (%f)'
        #       % (np.mean(warmstart_train_times),
        #          np.std(warmstart_train_times)))

        # del base_neural_net
        # gc.collect()

        # print('Warm-start hyperparameters:', warmstart_best_params)

        # print()

        # ---------------------------------------------------------------------

        print('-' * 80)
        print('*** Fine-tuning kernel function ***')
        print()

        output_train_metrics_filename \
            = os.path.join(finetune_output_dir, 'train',
                           '%s_%s_exp%d_%s_train_metrics_%s.txt'
                           % (full_estimator_name, dataset, experiment_idx,
                              validation_string, finetune_hyperparam_hash))
        output_best_hyperparam_filename \
            = os.path.join(finetune_output_dir, 'train',
                           '%s_%s_exp%d_%s_best_hyperparams_%s.pkl'
                           % (full_estimator_name, dataset, experiment_idx,
                              validation_string, finetune_hyperparam_hash))
        if not os.path.isfile(output_train_metrics_filename) or \
                not os.path.isfile(output_best_hyperparam_filename):
            print('Training...', flush=True)
            train_metrics_file = open(output_train_metrics_filename, 'w')
            best_hyperparams = {}

            min_loss = np.inf
            arg_min = None
            best_model_filename = None

            for hyperparam_idx, hyperparam in enumerate(finetune_hyperparams):
                squared_radius, n_layers, n_nodes, batch_size, \
                    alpha, sigma, n_durations, gamma, beta, \
                    min_kernel_weight, n_neighbors, batch_size, max_n_epochs, \
                    lr = hyperparam

                if alpha == 0 and sigma != sigma_range[0]:
                    continue
                if squared_radius == 0 and gamma > 0:
                    continue

                # seed different hyperparameters differently to prevent weird
                # behavior where a bad initial seed makes a specific model
                # always look terrible
                hyperparam_random_seed = finetune_method_random_seed \
                    + hyperparam_idx

                dataset_max_n_epochs = 'max_n_epochs_%s' % dataset
                if dataset_max_n_epochs in config[finetune_method_header]:
                    max_n_epochs = \
                        int(config[finetune_method_header][
                            dataset_max_n_epochs])

                tic = time.time()
                torch.manual_seed(hyperparam_random_seed)
                np.random.seed(hyperparam_random_seed)
                random.seed(hyperparam_random_seed)

                if squared_radius == 0:
                    raise ValueError('squared radius equal to 0 not supported')
                else:
                    base_neural_net = create_base_neural_net_with_hypersphere(
                        num_input_features,
                        [n_nodes for _ in range(n_layers)], squared_radius=squared_radius
                        ).to(device)
                # warmstart_checkpoint = torch.load(warmstart_model_filename)
                # base_neural_net.load_state_dict(warmstart_checkpoint['model_state_dict'])
                
                dkaj_model = DKAJ(base_neural_net, device=device, alpha=alpha, sigma=sigma, beta=beta, tau=np.sqrt(-np.log(min_kernel_weight)))
                dkaj_loss = dkaj_model.loss

                finetune_model_filename = \
                    os.path.join(
                        finetune_output_dir, 'models',
                        '%s_%s_exp%d_a%f_s%f_nd%d_g%f_b%f_mkw%f_sr%f_'
                        % (full_estimator_name, dataset, experiment_idx,
                           alpha, sigma, n_durations, gamma,
                           beta, min_kernel_weight, squared_radius)
                        +
                        'nn%d_nla%d_nno%d_bs%d_mnep%d_lr%f.pt'
                        % (n_neighbors, n_layers, n_nodes, batch_size,
                           max_n_epochs, lr))
                time_elapsed_filename = \
                    finetune_model_filename[:-3] + '_time.txt'
                epoch_time_elapsed_filename = \
                    finetune_model_filename[:-3] + '_epoch_times.txt'
                epoch_times = []

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
                time_grid_train_np = label_transform.cuts

                # Training and validation set after discretized times
                X_train = torch.tensor(X_train_np, dtype=torch.float32, device=device)
                Y_train = torch.tensor(Y_train_discrete_np, dtype=torch.int64, device=device)
                D_train = torch.tensor(D_train_discrete_np, dtype=torch.int32, device=device)
                train_data = list(zip(X_train, Y_train, D_train))

                X_val = torch.tensor(X_val_np, dtype=torch.float32, device=device)
                Y_val = torch.tensor(Y_val_discrete_np, dtype=torch.int64, device=device)
                D_val = torch.tensor(D_val_discrete_np, dtype=torch.int32, device=device)
                val_data = list(zip(X_val, Y_val, D_val))
                train_loader = DataLoader(train_data, batch_size, shuffle=True)  # shuffling for minibatch gradient descent
                val_loader = DataLoader(val_data, batch_size, shuffle=False)  # there is no need to shuffle the validation data

                optimizer = torch.optim.Adam(base_neural_net.parameters(), lr=lr)
                
                best_loss = np.inf
                for epoch_idx in range(max_n_epochs):
                    tic_ = time.time()
                    base_neural_net.train()
                    for X_batch, Y_batch, D_batch in train_loader:
                        neural_net_output = base_neural_net(X_batch)

                        # rank_mat = pair_rank_mat(Y_batch.cpu().numpy(), D_batch.cpu().numpy())
                        # rank_mat = torch.tensor(rank_mat, dtype=torch.int, device=device)

                        loss_batch = dkaj_loss(neural_net_output, Y_batch, D_batch)

                        optimizer.zero_grad()
                        loss_batch.backward()
                        optimizer.step()
                    epoch_train_time = time.time() - tic_
                    tic_ = time.time()
                    dkaj_model.training_data = (X_train_np.astype('float32'),
                                                (Y_train_discrete_np, D_train_discrete_np))
                    dkaj_model.train_embeddings = dkaj_model.predict(X_train_np.astype('float32'),
                                                                    batch_size=batch_size)
                    dkaj_model.duration_index = time_grid_train_np
                    dkaj_model.build_ANN_index()
                    epoch_train_postprocess_time = time.time() - tic_
                    tic_ = time.time()
                    cif_pred_all_events = dkaj_model.predict_cif(X_val_np, batch_size=batch_size, to_cpu=True, numpy=True) # (n_events, n_durations, n_patients)
                    ibs_all_events, ctd_all_events = [], []
                    for e_idx_minus_1, event in enumerate(events):
                        cif_val_pred_event = cif_pred_all_events[e_idx_minus_1]  # shape (n_durations, n_patients)
                        # Interpolation: evaluate CIF for each patient at all_eval_horizons
                        cif_interp = np.empty((len(all_eval_horizons), cif_val_pred_event.shape[1]))
                        for j in range(cif_val_pred_event.shape[1]):
                            cif_interp[:, j] = np.interp(
                                all_eval_horizons, 
                                dkaj_model.duration_index, 
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
                                        epoch_train_postprocess_time,
                                        epoch_val_time])
                    new_hyperparam = \
                        (squared_radius, n_layers, n_nodes, batch_size, 
                            alpha, sigma, n_durations,
                            gamma, beta, min_kernel_weight, n_neighbors,
                            batch_size, epoch_idx + 1, lr,
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
                            'train postprocess time %f sec(s)'
                            % epoch_train_postprocess_time,
                            '--',
                            'val time %f sec(s)' % epoch_val_time,
                            flush=True)
                    print(new_hyperparam, ':', val_loss, flush=True,
                            file=train_metrics_file)

                    if val_loss < best_loss:
                        best_loss = val_loss
                        wait_idx = 0
                        dkaj_model.save_net(finetune_model_filename)

                        if val_loss < min_loss:
                            min_loss = val_loss
                            arg_min = new_hyperparam
                            best_model_filename = \
                                finetune_model_filename
                    else:
                        wait_idx += 1
                        if patience > 0 and wait_idx >= patience:
                            break

                np.savetxt(epoch_time_elapsed_filename,
                           np.array(epoch_times))

                elapsed = time.time() - tic
                print('Time elapsed: %f second(s)' % elapsed, flush=True)
                np.savetxt(time_elapsed_filename,
                           np.array(elapsed).reshape(1, -1))

                del dkaj_model
                gc.collect()

            train_metrics_file.close()

            best_hyperparams['loss'] = (arg_min, min_loss)
            with open(output_best_hyperparam_filename, 'wb') as pickle_file:
                pickle.dump(best_hyperparams, pickle_file,
                            protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print('Loading previous validation results...', flush=True)
            with open(output_best_hyperparam_filename, 'rb') as pickle_file:
                best_hyperparams = pickle.load(pickle_file)
            arg_min, min_loss = best_hyperparams['loss']

        print('Best hyperparameters for minimizing loss:',
              arg_min, '-- achieves val %s' % model_selection_metric, '%.4f avg' % min_loss, flush=True)

        print()

        # ---------------------------------------------------------------------
        # Summary fine-tuning
        #

        squared_radius, n_layers, n_nodes, batch_size, \
            alpha, sigma, n_durations, gamma, beta, \
            min_kernel_weight, n_neighbors, batch_size, n_epochs, lr, seed \
            = arg_min

        tic = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if squared_radius == 0:
            raise ValueError('squared radius equal to 0 not supported')
        else:
            base_neural_net = create_base_neural_net_with_hypersphere(
                        num_input_features,
                        [n_nodes for _ in range(n_layers)], squared_radius=squared_radius
                        ).to(device)

        # warmstart_checkpoint = torch.load(warmstart_model_filename)
        # base_neural_net.load_state_dict(warmstart_checkpoint['model_state_dict'])
        dkaj_model = DKAJ(base_neural_net, device=device, alpha=alpha, sigma=sigma, beta=beta, tau=np.sqrt(-np.log(min_kernel_weight)))
        dkaj_model.net.train()

        max_n_epochs = int(config[finetune_method_header]['max_n_epochs'])
        dataset_max_n_epochs = 'max_n_epochs_%s' % dataset
        if dataset_max_n_epochs in config[finetune_method_header]:
            max_n_epochs = \
                int(config[finetune_method_header][
                    dataset_max_n_epochs])

        model_filename = \
            os.path.join(
                finetune_output_dir, 'models',
                '%s_%s_exp%d_a%f_s%f_nd%d_g%f_b%f_mkw%f_sr%f_nn%d_'
                % (full_estimator_name, dataset, experiment_idx,
                   alpha, sigma, n_durations, gamma, beta, min_kernel_weight,
                   squared_radius, n_neighbors)
                +
                'nla%d_nno%d_bs%d_mnep%d_lr%f.pt'
                % (n_layers, n_nodes, batch_size, max_n_epochs, lr))
        dkaj_model.load_net(model_filename)

        # recompute label transformer used for the base neural net
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
        time_grid_train_np = label_transform.cuts

        X_train = torch.tensor(X_train_np, dtype=torch.float32, device=device)
        Y_train = torch.tensor(Y_train_discrete_np, dtype=torch.int64, device=device)
        D_train = torch.tensor(D_train_discrete_np, dtype=torch.int32, device=device)
        train_data = list(zip(X_train, Y_train, D_train))

        X_val = torch.tensor(X_val_np, dtype=torch.float32, device=device)
        Y_val = torch.tensor(Y_val_discrete_np, dtype=torch.int64, device=device)
        D_val = torch.tensor(D_val_discrete_np, dtype=torch.int32, device=device)
        val_data = list(zip(X_val, Y_val, D_val))

        train_loader = DataLoader(train_data, batch_size, shuffle=True)  # shuffling for minibatch gradient descent
        val_loader = DataLoader(val_data, batch_size, shuffle=False)  # there is no need to shuffle the validation data
        
        dkaj_model.training_data = (X_train_np.astype('float32'),
                                    (Y_train_discrete_np, D_train_discrete_np))
        dkaj_model.train_embeddings = dkaj_model.predict(X_train_np.astype('float32'),
                                                        batch_size=batch_size)
        dkaj_model.duration_index = time_grid_train_np
        dkaj_model.build_ANN_index()

        time_elapsed_filename = model_filename[:-3] + '_time.txt'
        elapsed = float(np.loadtxt(time_elapsed_filename))
        print('Time elapsed (from previous fitting): %f second(s)'
              % elapsed, flush=True)

        if finetune_summaries and beta > 0:
            print('-' * 80)
            print('*** Fine-tuning exemplar summaries ***')
            print()


            summary_finetune_model_filename = model_filename[:-3] + \
                '_summary_finetune.pt'
            summary_finetune_arg_min_filename = \
                summary_finetune_model_filename[:-3] + '_arg_min.txt'
            summary_finetune_min_filename = \
                summary_finetune_model_filename[:-3] + '_min_loss.txt'
            summary_finetune_epoch_time_elapsed_filename = \
                summary_finetune_model_filename[:-3] + '_epoch_times.txt'
            summary_finetune_metrics_filename = \
                summary_finetune_model_filename[:-3] + '_metrics.txt'
            if not os.path.isfile(summary_finetune_model_filename) \
                    or not os.path.isfile(
                        summary_finetune_arg_min_filename) \
                    or not os.path.isfile(
                        summary_finetune_min_filename) \
                    or not os.path.isfile(
                        summary_finetune_epoch_time_elapsed_filename):
                finetune_metrics_file = \
                    open(summary_finetune_metrics_filename, 'w')
                arg_min_finetune = None
                finetune_min_loss = np.inf
                epoch_times = []
                for finetune_lr_idx, finetune_lr in \
                        enumerate(sumtune_lr_range):
                    torch.manual_seed(finetune_method_random_seed)
                    np.random.seed(finetune_method_random_seed)
                    random.seed(finetune_method_random_seed)

                    init_summary_functions = dkaj_model.get_summary_functions()
                    summary_finetune_net = \
                        DKAJSummary(dkaj_model, init_summary_functions[0], init_summary_functions[1])
                    summary_finetune_loss = DKAJSummaryLoss(alpha, sigma)
                    optimizer = torch.optim.Adam(summary_finetune_net.parameters(),
                                                 lr=finetune_lr)
                    best_loss = np.inf
                    # record the preprocessing time
                    if finetune_lr_idx == 0:
                        epoch_times.append(
                            [time.time() - tic, 0., 0,])
                    for epoch_idx in range(max_n_epochs):
                        tic_ = time.time()
                        summary_finetune_net.train()
                        for X_batch, Y_batch, D_batch in train_loader:
                            embedding_vectors = base_neural_net(X_batch)
                            neural_net_output = summary_finetune_net(embedding_vectors)
                            loss_batch = summary_finetune_loss(neural_net_output[0],
                                                            neural_net_output[1],
                                                            Y_batch, D_batch)

                            optimizer.zero_grad()
                            loss_batch.backward()
                            optimizer.step()
                        epoch_train_time = time.time() - tic_

                        tic_ = time.time()
                        
                        exemplar_event_counts, exemplar_at_risk_counts, baseline_event_counts, baseline_at_risk_counts = \
                            summary_finetune_net.get_exemplar_summary_functions_baseline_event_at_risk_counts()
                        dkaj_model.load_summary_functions(exemplar_event_counts,
                                                          exemplar_at_risk_counts,
                                                          baseline_event_counts,
                                                          baseline_at_risk_counts)
                        epoch_train_postprocess_time = time.time() - tic_
                        tic_ = time.time()
                        cif_pred_all_events = dkaj_model.predict_cif(X_val_np, batch_size=batch_size, to_cpu=True, numpy=True) # (n_events, n_durations, n_patients)
                        ibs_all_events, ctd_all_events = [], []
                        for e_idx_minus_1, event in enumerate(events):
                            cif_val_pred_event = cif_pred_all_events[e_idx_minus_1]  # shape (n_durations, n_patients)
                            # Interpolation: evaluate CIF for each patient at all_eval_horizons
                            cif_interp = np.empty((len(all_eval_horizons), cif_val_pred_event.shape[1]))
                            for j in range(cif_val_pred_event.shape[1]):
                                cif_interp[:, j] = np.interp(
                                    all_eval_horizons, 
                                    dkaj_model.duration_index, 
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
                                            epoch_train_postprocess_time,
                                            epoch_val_time])
                        print((epoch_idx + 1, finetune_lr),
                                '--',
                                'val avg IBS %.4f' % avg_ibs,
                                '--',
                                'val avg Ctd %.4f' % avg_ctd,
                                '--',
                                'train time %f sec(s)'
                                % epoch_train_time,
                                '--',
                                'train postprocess time %f sec(s)'
                                % epoch_train_postprocess_time,
                                '--',
                                'val time %f sec(s)' % epoch_val_time,
                                flush=True)
                        print(finetune_lr, ':', val_loss, flush=True,
                                file=finetune_metrics_file)

                        if val_loss < best_loss:
                            best_loss = val_loss
                            wait_idx = 0

                            if val_loss < finetune_min_loss:
                                finetune_min_loss = val_loss
                                arg_min_finetune = (epoch_idx + 1,
                                                    finetune_lr)
                                torch.save({
                                    'model_state_dict': summary_finetune_net.state_dict(),
                                    'val_loss': val_loss,
                                    'finetune_lr': finetune_lr
                                    }, summary_finetune_model_filename)
                        else:
                            wait_idx += 1
                            if patience > 0 and wait_idx >= patience:
                                break


                    summary_finetune_net = summary_finetune_net.cpu()
                    del summary_finetune_net
                    torch.cuda.empty_cache()

                np.savetxt(summary_finetune_arg_min_filename,
                           np.array(arg_min_finetune).reshape(1, -1))
                np.savetxt(summary_finetune_min_filename,
                           np.array(finetune_min_loss).reshape(1, -1))
                np.savetxt(summary_finetune_epoch_time_elapsed_filename,
                           np.array(epoch_times))

            arg_min_finetune = \
                np.loadtxt(summary_finetune_arg_min_filename).flatten()
            arg_min_finetune_n_epochs, arg_min_finetune_lr = \
                arg_min_finetune
            arg_min_finetune_n_epochs = int(arg_min_finetune_n_epochs)
            finetune_min_loss = \
                float(np.loadtxt(summary_finetune_min_filename))

            torch.manual_seed(finetune_method_random_seed)
            np.random.seed(finetune_method_random_seed)
            random.seed(finetune_method_random_seed)

            init_summary_functions = dkaj_model.get_summary_functions()
            if finetune_min_loss < min_loss:
                
                summary_finetune_net = \
                        DKAJSummary(dkaj_model, init_summary_functions[0], init_summary_functions[1])
                summary_finetune_checkpoint = torch.load(summary_finetune_model_filename)
                summary_finetune_net.load_state_dict(summary_finetune_checkpoint['model_state_dict'])
                exemplar_event_counts, exemplar_at_risk_counts, baseline_event_counts, baseline_at_risk_counts = \
                    summary_finetune_net.get_exemplar_summary_functions_baseline_event_at_risk_counts()
                dkaj_model.load_summary_functions(exemplar_event_counts,
                                                exemplar_at_risk_counts,
                                                baseline_event_counts,
                                                baseline_at_risk_counts)
            else:
                print('*** Warning: Summary fine-tuning did not result'
                      ' in lower validation loss')
                dkaj_model.load_summary_functions(init_summary_functions[0],
                                                  init_summary_functions[1])

            epoch_times = np.loadtxt(
                summary_finetune_epoch_time_elapsed_filename)
            print('Best summary fine-tuning hyperparameters:',
                  (arg_min_finetune_n_epochs, arg_min_finetune_lr),
                  '-- val loss:', finetune_min_loss)
            print('Time elapsed (from previous fitting): %f second(s)'
                  % epoch_times.sum())

        print()

        print('Saving files that can be used to help make visualizations...',
              flush=True)
        visualization_dir = \
            os.path.join(
                finetune_output_dir, 'visualization',
                '%s_%s_exp%d_a%f_s%f_nd%d_g%f_b%f_mkw%f_sr%f_nn%d_'
                % (full_estimator_name, dataset, experiment_idx,
                   alpha, sigma, n_durations, gamma, beta,
                   min_kernel_weight, squared_radius, n_neighbors)
                +
                'nla%d_nno%d_bs%d_nep%d_lr%f'
                % (n_layers, n_nodes, batch_size, n_epochs, lr))
        if finetune_summaries:
            visualization_dir = visualization_dir + '_sft'

        os.makedirs(visualization_dir, exist_ok=True)
        np.save(os.path.join(visualization_dir, 'train_raw_input.npy'),
                X_train_raw_np)
        np.save(os.path.join(visualization_dir, 'train_labels.npy'),
                np.stack((Y_train_np, D_train_np), axis=1))
        np.save(os.path.join(visualization_dir, 'duration_index.npy'),
                dkaj_model.duration_index)
        np.save(os.path.join(visualization_dir, 'train_embeddings.npy'),
                dkaj_model.train_embeddings)
        np.save(os.path.join(visualization_dir, 'exemplar_assignments.npy'),
                np.array(dkaj_model.exemplar_assignments, dtype=object))
        if dkaj_model.baseline_event_counts is not None:
            np.save(os.path.join(visualization_dir,
                                 'baseline_event_counts.npy'),
                    dkaj_model.baseline_event_counts)
        if dkaj_model.baseline_at_risk_counts is not None:
            np.save(os.path.join(visualization_dir,
                                 'baseline_at_risk_counts.npy'),
                    dkaj_model.baseline_at_risk_counts)

        print()

        # ---------------------------------------------------------------------
        # Test set prediction
        #

        print('Testing...', flush=True)
        X_test_np = apply_preprocessor(X_test_raw_np, preprocessor)
        X_test_np = X_test_np.astype('float32')
        Y_test_np = Y_test_np.astype('float32')
        D_test_np = D_test_np.astype('float32')
        
        cif_test_pred_all_events = dkaj_model.predict_cif(X_test_np, batch_size=batch_size, to_cpu=True, numpy=True) # shape (n_events, n_durations, n_patients)
        test_eval_brier_scores_all_events, test_ibs_all_events = [], []
        test_cindex_td_all_events, test_eval_cindex_cr_all_events = [], []
        for e_idx_minus_1, event in enumerate(events):
            cif_test_pred_event = cif_test_pred_all_events[e_idx_minus_1] # shape (n_durations, n_patients)
            # Interpolation: evaluate CIF for each patient at all_eval_horizons with linear interpolation
            cif_interp = np.empty((len(all_eval_horizons), cif_test_pred_event.shape[1]))
            for j in range(cif_test_pred_event.shape[1]):  # loop over patients
                cif_interp[:, j] = np.interp(
                    all_eval_horizons,           # new evaluation times
                    dkaj_model.duration_index,        # original grid
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

        del dkaj_model
        gc.collect()

        test_csv_writer.writerow(
            [dataset, experiment_idx, finetune_estimator_name] + test_set_metrics)
        output_test_table_file.flush()

        print()
        print()

output_test_table_file.close()