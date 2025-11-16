import hnswlib
import numba
import os
import numpy as np
import pickle
import pycox
import torch
import torchtuples as tt
from dsm import DeepSurvivalMachines
from nfg import NeuralFineGray
from desurv import DeSurv
from hazardous import SurvivalBoost
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.validation import check_array

import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from torch import Tensor
from typing import Optional

from multiprocessing import Pool, shared_memory

torch.manual_seed(0)


class CauseSpecificNet(torch.nn.Module):
    """Network structure similar to the DeepHit paper, but without the residual
    connections (for simplicity).
    """
    def __init__(self, in_features, num_nodes_shared, num_nodes_indiv, num_risks,
                 out_features, batch_norm=True, dropout=None):
        super().__init__()
        self.shared_net = tt.practical.MLPVanilla(
            in_features, num_nodes_shared[:-1], num_nodes_shared[-1],
            batch_norm, dropout,
        )
        self.risk_nets = torch.nn.ModuleList()
        for _ in range(num_risks):
            net = tt.practical.MLPVanilla(
                num_nodes_shared[-1], num_nodes_indiv, out_features,
                batch_norm, dropout,
            )
            self.risk_nets.append(net)

    def forward(self, input):
        out = self.shared_net(input)
        out = [net(out) for net in self.risk_nets]
        out = torch.stack(out, dim=1)
        return out
    

class Model(tt.Model):
    """
    Although torchtuples already supports fitting a single epoch at a time,
    there's a bit of an overhead. This version aims to be a little faster
    albeit with less features (e.g., no callbacks, no verbose output).
    """
    def fit_dataloader(self, dataloader, epochs=1, callbacks=None,
                       verbose=False, metrics=None, val_dataloader=None):
        if 'fit_info' not in self.__dict__:
            self._setup_train_info(dataloader)
            self.metrics = self._setup_metrics(metrics)
            self.log.verbose = False
            self.val_metrics.dataloader = val_dataloader

        for _ in range(epochs):
            for data in dataloader:
                self.optimizer.zero_grad()
                if data[0].size(0) > 1:
                    self.batch_metrics = self.compute_metrics(data,
                                                              self.metrics)
                    self.batch_loss = self.batch_metrics['loss']
                    if self.batch_loss.grad_fn is not None:
                        self.batch_loss.backward()
                        self.optimizer.step()


class DeepHit(pycox.models.DeepHit):
    """
    We modify DeepHit in the same way we modify Model
    """
    def fit_dataloader(self, dataloader, epochs=1, callbacks=None,
                       verbose=False, metrics=None, val_dataloader=None):
        if 'fit_info' not in self.__dict__:
            self._setup_train_info(dataloader)
            self.metrics = self._setup_metrics(metrics)
            self.log.verbose = False
            self.val_metrics.dataloader = val_dataloader

        for _ in range(epochs):
            for data in dataloader:
                self.optimizer.zero_grad()
                self.batch_metrics = self.compute_metrics(data, self.metrics)
                self.batch_loss = self.batch_metrics['loss']
                if self.batch_loss.grad_fn is not None:
                    self.batch_loss.backward()
                    self.optimizer.step()


class DeepSurvivalMachinesWrap(DeepSurvivalMachines):
    def save_net(self, filename):
        torch.save(self.torch_model.state_dict(), filename)

    def load_net(self, filename, x=None, t=None, e=None):
        """
        Load model weights from a .pt file.
        You must call fit with data to initialize torch_model before loading weights, 
        unless torch_model is already initialized.
        Args:
            filename (str): Path to the .pt file.
            x, t, e: Optional. If the model is not fitted yet, you must provide dummy data to initialize.
        """
        if not hasattr(self, 'torch_model') or not self.fitted:
            if x is None or t is None or e is None:
                raise ValueError("Provide x, t, e to initialize the model before loading weights.")
            # Fit on dummy data to initialize torch_model
            self.fit(x, t, e, iters=1)
        self.torch_model.load_state_dict(torch.load(filename))
        self.fitted = True


class NeuralFineGrayWrap(NeuralFineGray):
    def save_net(self, filename):
        torch.save(self.torch_model.state_dict(), filename)

    def load_net(self, filename, x=None, t=None, e=None):
        """
        Load model weights from a .pt file.
        You must call fit with data to initialize torch_model before loading weights, 
        unless torch_model is already initialized.
        Args:
            filename (str): Path to the .pt file.
            x, t, e: Optional. If the model is not fitted yet, you must provide dummy data to initialize.
        """
        if not hasattr(self, 'torch_model') or not self.fitted:
            if x is None or t is None or e is None:
                raise ValueError("Provide x, t, e to initialize the model before loading weights.")
            # Fit on dummy data to initialize torch_model
            self.fit(x, t, e, n_iter=1)
        self.torch_model.load_state_dict(torch.load(filename))
        self.fitted = True


class DeSurvWrap(DeSurv):
    def save_net(self, filename):
        torch.save(self.torch_model.state_dict(), filename)

    def load_net(self, filename, x=None, t=None, e=None):
        """
        Load model weights from a .pt file.
        You must call fit with data to initialize torch_model before loading weights, 
        unless torch_model is already initialized.
        Args:
            filename (str): Path to the .pt file.
            x, t, e: Optional. If the model is not fitted yet, you must provide dummy data to initialize.
        """
        if not hasattr(self, 'torch_model') or not self.fitted:
            if x is None or t is None or e is None:
                raise ValueError("Provide x, t, e to initialize the model before loading weights.")
            # Fit on dummy data to initialize torch_model
            self.fit(x, t, e, n_iter=1)
        self.torch_model.load_state_dict(torch.load(filename))
        self.fitted = True


class SurvivalBoostWrap(SurvivalBoost):
    def _build_base_estimator(self):
        '''
        This is just to patch the _build_base_estimator function to use the
        manually specified `random_state` (at the time of writing, this appears
        to be a bug so that specifying `random_state` in constructing
        `SurvivalBoost` does not result in deterministic results)
        '''
        return HistGradientBoostingClassifier(
            loss="log_loss",
            max_iter=1,
            warm_start=True,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
            )
    
    def save_model(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)


def hist_gradient_boosting_classifier_apply(estimator, X, n_threads=1):
    # X = estimator._preprocess_X(X, reset=False)
    X = check_array(X, dtype=[np.float32, np.float64], force_all_finite=True)

    (
        known_cat_bitsets,
        f_idx_map,
    ) = estimator._bin_mapper.make_known_categories_bitsets()

    all_trees = [[tree.nodes # tree.__getstate__()['nodes']
                  for tree in iter_predictors]
                 for iter_predictors in estimator._predictors]

    all_raw_left_cat_bitsets = \
        [[tree.raw_left_cat_bitsets # tree.__getstate__()['raw_left_cat_bitsets']
          for tree in iter_predictors]
         for iter_predictors in estimator._predictors]
    
    shm_X = shared_memory.SharedMemory(create=True,
                                       size=(X.dtype.itemsize *
                                             X.shape[0] *
                                             X.shape[1]))
    shm_X_data = np.ndarray(X.shape, dtype=X.dtype, buffer=shm_X.buf)
    shm_X_data[:, :] = X[:, :]

    with Pool(n_threads) as pool:
        results = pool.starmap(apply,
                               [(shm_X.name, X.shape, X.dtype, idx,
                                 all_trees, all_raw_left_cat_bitsets,
                                 known_cat_bitsets, f_idx_map)
                                for idx in range(X.shape[0])])

    shm_X.unlink()
    return np.array(results)

def apply(shm_X_name, X_shape, X_dtype, idx, all_trees,
          all_raw_left_cat_bitsets, known_cat_bitsets, f_idx_map):
    shm_X = shared_memory.SharedMemory(name=shm_X_name)
    x = np.ndarray(X_shape, dtype=X_dtype, buffer=shm_X.buf)[idx]

    result = np.zeros((len(all_trees), len(all_trees[0])), dtype=np.int64)

    for iter_idx, iter_predictors in enumerate(all_trees):
        for tree_idx, tree in enumerate(iter_predictors):
            node_idx = 0  # start at root node
            while True:
                node = tree[node_idx]
                raw_left_cat_bitsets = all_raw_left_cat_bitsets[iter_idx][tree_idx]
                if node['is_leaf']:
                    result[iter_idx, tree_idx] = node_idx
                    break
                val = x[node['feature_idx']]
                if np.isnan(val):
                    node_idx = node['left'] if node['missing_go_left'] else node['right']
                elif node['is_categorical']:
                    if val < 0:
                        node_idx = node['left'] if node['missing_go_left'] else node['right']
                    elif (raw_left_cat_bitsets[node['bitset_idx'],
                                               val // 32] >> (val % 32)) & 1:
                        node_idx = node['left']
                    elif (known_cat_bitsets[f_idx_map[node['feature_idx']],
                                            val // 32] >> (val % 32)) & 1:
                        node_idx = node['right']
                    else:
                        node_idx = node['left'] if node['missing_go_left'] else node['right']
                elif val <= node['num_threshold']:
                    node_idx = node['left']
                else:
                    node_idx = node['right']

    return result


def tuna_loss(neural_net_output, leaves_batch, device='cuda'):
    predicted_kernel_matrix = \
        (-symmetric_squared_pairwise_distances(neural_net_output)).exp() \
         - torch.eye(neural_net_output.size(0), device=device)
    target_kernel_matrix = \
        (1. - torch.cdist(leaves_batch, leaves_batch, 0) / leaves_batch.size(1)).view(-1)
    return F.mse_loss(predicted_kernel_matrix.view(-1), target_kernel_matrix.view(-1))


@numba.njit
def _pair_rank_mat_cr(mat, idx_durations, events, event_idx,
                      dtype='float32'):
    n = len(idx_durations)
    for i in range(n):
        dur_i = idx_durations[i]
        ev_i = events[i]
        if ev_i != event_idx:
            continue
        for j in range(n):
            dur_j = idx_durations[j]
            ev_j = events[j]
            if (dur_i < dur_j) or ((dur_i == dur_j) and (ev_j == 0)):
                mat[i, j] = 1
    return mat


class Hypersphere(nn.Module):
    def __init__(self, squared_radius: Optional[float] = 1.) -> None:
        super(Hypersphere, self).__init__()
        self.squared_radius = squared_radius

    def forward(self, input: Tensor) -> Tensor:
        return F.normalize(input, dim=1) / np.sqrt(self.squared_radius)
    

def create_base_neural_net_with_hypersphere(inputdim, layers, activation='ReLU', squared_radius=0.1):
    if activation == 'ReLU6':
        act = nn.ReLU6()
    elif activation == 'ReLU':
        act = nn.ReLU()
    elif activation == 'SeLU':
        act = nn.SELU()

    modules = []
    prevdim = inputdim

    for hidden in layers:
        modules.append(nn.Linear(prevdim, hidden, bias=True))
        modules.append(act)
        prevdim = hidden

    modules.append(Hypersphere(squared_radius=squared_radius))
    return nn.Sequential(*modules)


# symmetric squared Euclidean distance calculation from:
# https://discuss.pytorch.org/t/build-your-own-loss-function-in-pytorch/235/7
def symmetric_squared_pairwise_distances(x):
    r = torch.mm(x, x.t())
    diag = r.diag().unsqueeze(0)
    diag = diag.expand_as(r)
    return diag + diag.t() - 2*r


class DKAJLoss(torch.nn.Module):
    def __init__(self, alpha, sigma, loo=True):
        super(DKAJLoss, self).__init__()
        self.alpha = alpha
        self.sigma = sigma
        self.loo = loo
    def forward(self, phi: Tensor, idx_durations: Tensor,
                events: Tensor) -> Tensor:
        return nll_kernel_hazard_cr(phi, idx_durations, events, self.alpha,
                                    self.sigma, loo=self.loo)


def nll_kernel_hazard_cr(phi: Tensor, idx_durations: Tensor, events: Tensor,
                         alpha: float, sigma: float,
                         reduction: str = 'mean',
                         multiply_by_surv_at_prev_time_step: bool = True,
                         loo: bool = True) -> Tensor:
    if events.dtype is not torch.float:
        events = events.float()

    batch_size = phi.size(0)
    n_risks = phi.size(1)
    num_durations = idx_durations.max().item() + 1

    rank_matrices = []
    if alpha < 1:
        idx_durations_np = idx_durations.detach().cpu().numpy().ravel()
        events_np = events.detach().cpu().numpy().ravel()
        for risk_idx in range(n_risks):
            rank_mat_np = np.zeros((batch_size, batch_size),
                                   dtype='float32')
            rank_matrices.append(
                torch.tensor(
                    _pair_rank_mat_cr(
                        rank_mat_np, idx_durations_np, events_np,
                        risk_idx + 1, 'float32'
                    ),
                    device=phi.device
                )
            )

    idx_durations = idx_durations.view(-1, 1)
    events = events.view(-1, 1) - 1

    # compute kernel matrix
    kernel_matrix = (-symmetric_squared_pairwise_distances(phi)).exp()
    if loo:
        weights_loo = kernel_matrix - torch.eye(batch_size, device=phi.device)
    else:
        weights_loo = kernel_matrix # for ablation study

    obs_time_matrix = torch.zeros((batch_size, num_durations),
                                   dtype=torch.float,
                                   device=phi.device).scatter(1, idx_durations,
                                                              1)
    B = ((obs_time_matrix.flip(1)).cumsum(1)).flip(1)

    # bin weights in the same time index together (only for columns)
    weights_loo_discretized = torch.matmul(weights_loo, obs_time_matrix)

    num_at_risk = ((weights_loo_discretized.flip(1)).cumsum(1)).flip(1) + 1e-12

    nll_loss = torch.zeros(batch_size, dtype=torch.float32, device=phi.device)
    event_specific_hazards = []
    overall_numer = 0
    for risk_idx in range(n_risks):
        A = torch.zeros((batch_size, num_durations), dtype=torch.float,
                        device=phi.device).scatter(1, idx_durations,
            1*(events == risk_idx).to(dtype=torch.float))

        # kernel hazard function calculation
        num_events = torch.matmul(weights_loo, A)
        hazards = torch.clamp(num_events / num_at_risk, 1e-12, 1. - 1e-12)

        if not torch.all((hazards >= 0) & (hazards <= 1)):
            # weird corner case
            return torch.tensor(np.inf, dtype=torch.float32)

        nll_loss -= (A * torch.log(hazards) - B * hazards).sum(1)

        event_specific_hazards.append(hazards)
        overall_numer += num_events

    if alpha < 1:
        weighted_overall_hazards = torch.clamp(overall_numer / num_at_risk,
                                               1e-12, 1. - 1e-12)
        surv = \
            (1 - weighted_overall_hazards).add(1e-7).log().cumsum(1).exp()
        rank_loss = 0
        if multiply_by_surv_at_prev_time_step:
            surv_shifted = torch.roll(surv, shifts=1, dims=1)
            surv_shifted[:, 0] = 1.
        else:
            surv_shifted = surv
        for risk_idx in range(n_risks):
            cif = (event_specific_hazards[risk_idx] * surv_shifted).cumsum(1)
            event_matrix = torch.zeros((batch_size, num_durations),
                                       dtype=torch.float,
                                       device=phi.device).scatter(
                1, idx_durations, 1.*(events == risk_idx))
            A = cif.matmul(event_matrix.transpose(0, 1))
            diag_A = A.diag().view(1, -1)
            ones = torch.ones((batch_size, 1), device=phi.device)
            differences = (A - ones.matmul(diag_A)).transpose(0, 1)
            rank_loss += \
                (rank_matrices[risk_idx] *
                 torch.exp(differences / sigma)).mean(1, keepdim=True)

        return alpha * pycox.models.loss._reduction(nll_loss, reduction) + \
            (1 - alpha) * pycox.models.loss._reduction(rank_loss, reduction)
    else:
        return pycox.models.loss._reduction(nll_loss, reduction)


class KernelModel(Model):
    def __init__(self, net, loss=None, optimizer=None, device=None,
                 beta=0., tau=np.sqrt(-np.log(1e-4)), brute_force=False,
                 max_n_neighbors=256, dkn_max_n_neighbors=256,
                 ann_random_seed=2591188910,
                 ann_deterministic_construction=True):
        super(KernelModel, self).__init__(net, loss, optimizer, device)

        self.beta = beta
        self.tau = tau
        self.brute_force = brute_force
        self.n_neighbors = max_n_neighbors
        self.dkn_n_neighbors = dkn_max_n_neighbors
        self.ann_random_seed = ann_random_seed
        self.ann_deterministic_construction = ann_deterministic_construction

        self.training_data = None
        self.train_embeddings = None
        self.ann_index = None

    def build_ANN_index(self):
        if self.training_data is None or self.train_embeddings is None:
            raise Exception(
                'Set `self.training_data` and `self.train_embeddings` first.')

        train_embeddings = self.train_embeddings
        if type(train_embeddings) == torch.Tensor:
            train_embeddings = train_embeddings.detach().cpu().numpy()
        n_train = train_embeddings.shape[0]

        self.average_label = self.training_data[1].mean(axis=0)

        if self.beta == 0:
            ann_index = hnswlib.Index(space='l2',
                                      dim=train_embeddings.shape[1])
            if self.ann_deterministic_construction:
                ann_index.set_num_threads(1)
            else:
                ann_index.set_num_threads(os.cpu_count())
            ann_index.init_index(max_elements=n_train,
                                 ef_construction=2*self.n_neighbors,
                                 M=48,
                                 random_seed=self.ann_random_seed)
            ann_index.add_items(train_embeddings, np.arange(n_train))
            ann_index.set_ef(min(2*self.n_neighbors, n_train))

            self.ann_index = ann_index

        else:
            eps = self.beta * self.tau
            eps_squared = eps * eps

            # build epsilon-net
            ann_index = \
                hnswlib.Index(space='l2', dim=train_embeddings.shape[1])
            ann_index.set_num_threads(1)
            ann_index.init_index(max_elements=n_train,
                                 ef_construction=2*self.dkn_n_neighbors,
                                 M=48,
                                 random_seed=self.ann_random_seed)
            ann_index.add_items([train_embeddings[0]], [0])
            assignments = [[0]]
            for idx in range(1, n_train):
                pt = train_embeddings[idx]
                labels, sq_dists = ann_index.knn_query([pt], k=1)
                nearest_exemplar_idx = labels[0, 0]
                sq_dist = sq_dists[0, 0]
                if sq_dist > eps_squared:
                    ann_index.add_items([pt], [len(assignments)])
                    assignments.append([idx])
                else:
                    assignments[nearest_exemplar_idx].append(idx)
            n_exemplars = len(assignments)
            if n_exemplars < self.dkn_n_neighbors:
                self.dkn_n_neighbors = n_exemplars
            ann_index.set_ef(min(2*self.dkn_n_neighbors, n_exemplars))

            self.ann_index = ann_index

            # compute summary statistics
            self.exemplar_assignments = [np.array(_) for _ in assignments]
            self.exemplar_sizes = np.array([len(_) for _ in assignments])
            self.exemplar_labels = \
                np.array([self.training_data[1][_].sum(axis=0)
                          for _ in assignments])

    def predict_classification(self, input, batch_size=256, numpy=None,
                               eval_=True, grads=False, to_cpu=False,
                               num_workers=0, is_dataloader=None, func=None,
                               **kwargs):
        if self.training_data is None or self.train_embeddings is None:
            raise Exception(
                'Set `self.training_data` and `self.train_embeddings` first.')

        test_embeddings = self.predict(input, batch_size, numpy, eval_,
                                       grads, to_cpu, num_workers,
                                       is_dataloader, func, **kwargs)
        if type(test_embeddings) == torch.Tensor:
            with torch.no_grad():
                test_embeddings = test_embeddings.detach().cpu().numpy()

        tau_squared = self.tau ** 2
        if self.brute_force:
            if self.beta == 0:
                sq_dists = cdist(test_embeddings, self.train_embeddings,
                                 'sqeuclidean')
            else:
                exemplars = np.array([_[0] for _ in self.exemplar_assignments],
                                     dtype=np.int64)
                sq_dists = cdist(test_embeddings,
                                 self.train_embeddings[exemplars],
                                 'sqeuclidean')

            nan_mask = np.isnan(sq_dists)
            if np.any(nan_mask):
                sq_dists[nan_mask] = 0.

            weights = np.exp(-sq_dists) * (sq_dists <= tau_squared)

        else:
            if self.beta == 0:
                weights_shape = ((test_embeddings.shape[0],
                                  self.train_embeddings.shape[0]))
            else:
                weights_shape = ((test_embeddings.shape[0],
                                  len(self.exemplar_assignments)))

            if num_workers <= 0:
                n_threads = os.cpu_count()
            else:
                n_threads = num_workers

            if self.beta == 0:
                k = self.n_neighbors
            else:
                k = self.dkn_n_neighbors
            labels, sq_dists = self.ann_index.knn_query(test_embeddings, k=k,
                                                        num_threads=n_threads)

            nan_mask = np.isnan(sq_dists)
            if np.any(nan_mask):
                sq_dists[nan_mask] = 0.

            row_col_pairs = \
                np.array([(row, col)
                          for row, cols in enumerate(labels) for col in cols],
                         dtype=np.int64)
            rows = row_col_pairs[:, 0]
            cols = row_col_pairs[:, 1]
            sq_dists_flat = sq_dists.flatten()
            weights = csr_matrix((np.exp(-sq_dists_flat)
                                  * (sq_dists_flat <= tau_squared),
                                 (rows, cols)), shape=weights_shape)

        # what the paper calls gamma * n
        gamma_n = np.exp(-tau_squared) / self.train_embeddings.shape[0]

        if self.beta == 0:
            unnormalized = weights.dot(self.training_data[1]) \
                + gamma_n * self.average_label
        else:
            unnormalized = weights.dot(self.exemplar_labels) \
                + gamma_n * self.average_label
        return unnormalized

    def predict_regression(self, input, batch_size=256, numpy=None,
                           eval_=True, grads=False, to_cpu=False,
                           num_workers=0, is_dataloader=None, func=None,
                           **kwargs):
        if self.training_data is None or self.train_embeddings is None:
            raise Exception(
                'Set `self.training_data` and `self.train_embeddings` first.')

        test_embeddings = self.predict(input, batch_size, numpy, eval_,
                                       grads, to_cpu, num_workers,
                                       is_dataloader, func, **kwargs)
        if type(test_embeddings) == torch.Tensor:
            with torch.no_grad():
                test_embeddings = test_embeddings.detach().cpu().numpy()

        tau_squared = self.tau ** 2
        if self.brute_force:
            if self.beta == 0:
                sq_dists = cdist(test_embeddings, self.train_embeddings,
                                 'sqeuclidean')
            else:
                exemplars = np.array([_[0] for _ in self.exemplar_assignments],
                                     dtype=np.int64)
                sq_dists = cdist(test_embeddings,
                                 self.train_embeddings[exemplars],
                                 'sqeuclidean')

            nan_mask = np.isnan(sq_dists)
            if np.any(nan_mask):
                sq_dists[nan_mask] = 0.

            weights = np.exp(-sq_dists) * (sq_dists <= tau_squared)

        else:
            if self.beta == 0:
                weights_shape = ((test_embeddings.shape[0],
                                  self.train_embeddings.shape[0]))
            else:
                weights_shape = ((test_embeddings.shape[0],
                                  len(self.exemplar_assignments)))

            if num_workers <= 0:
                n_threads = os.cpu_count()
            else:
                n_threads = num_workers

            if self.beta == 0:
                k = self.n_neighbors
            else:
                k = self.dkn_n_neighbors
            labels, sq_dists = self.ann_index.knn_query(test_embeddings, k=k,
                                                        num_threads=n_threads)

            nan_mask = np.isnan(sq_dists)
            if np.any(nan_mask):
                sq_dists[nan_mask] = 0.

            row_col_pairs = \
                np.array([(row, col)
                          for row, cols in enumerate(labels) for col in cols],
                         dtype=np.int64)
            rows = row_col_pairs[:, 0]
            cols = row_col_pairs[:, 1]
            sq_dists_flat = sq_dists.flatten()
            weights = csr_matrix((np.exp(-sq_dists_flat)
                                  * (sq_dists_flat <= tau_squared),
                                 (rows, cols)), shape=weights_shape)

        # what the paper calls gamma * n
        gamma_n = np.exp(-tau_squared) / self.train_embeddings.shape[0]

        if self.beta == 0:
            unnormalized = weights.dot(self.training_data[1]) \
                + gamma_n * self.average_label
            denom = weights.sum(axis=1) + gamma_n
        else:
            unnormalized = weights.dot(self.exemplar_labels) \
                + gamma_n * self.average_label
            denom = weights.dot(self.exemplar_sizes) + gamma_n
        if type(denom) != np.ndarray:
            denom = np.asarray(denom).ravel()
        return unnormalized / denom



class DKAJ(KernelModel):
    """
    Neural kernel survival analysis estimator; paired with a deep net as the
    base neural net/encoder, then one obtains a deep kernel survival analysis
    model
    """
    def __init__(self, net, optimizer=None, device=None, loss=None,
                 alpha=0.0, sigma=1.0, beta=0.0,
                 tau=np.sqrt(-np.log(1e-4)), brute_force=False,
                 max_n_neighbors=256, dkn_max_n_neighbors=256,
                 ann_random_seed=2591188910,
                 ann_deterministic_construction=True,
                 loo=True):
        ann_det_con = ann_deterministic_construction
        if loss is None:
            loss = DKAJLoss(alpha, sigma, loo=loo)
        super(DKAJ, self).__init__(net, loss=loss, optimizer=optimizer,
                                   device=device, beta=beta, tau=tau,
                                   brute_force=brute_force,
                                   max_n_neighbors=max_n_neighbors,
                                   dkn_max_n_neighbors=dkn_max_n_neighbors,
                                   ann_random_seed=ann_random_seed,
                                   ann_deterministic_construction=ann_det_con)
        self.n_risks = None

    def build_ANN_index(self):
        if self.training_data is None or self.train_embeddings is None:
            raise Exception(
                'Set `self.training_data` and `self.train_embeddings` first.')

        train_embeddings = self.train_embeddings
        if type(train_embeddings) == torch.Tensor:
            train_embeddings = train_embeddings.detach().cpu().numpy()
        n_train = train_embeddings.shape[0]
        num_durations = len(self.duration_index)
        idx_durations, events = self.training_data[1]

        n_risks = int(events.max())
        self.n_risks = n_risks

        self.overall_event_counts, self.overall_at_risk_counts = \
            survival_summarize_cr(idx_durations, events, num_durations,
                                  n_risks)

        self.overall_hazard = np.zeros((num_durations, n_risks))
        for risk_idx in range(n_risks):
            self.overall_hazard[:, risk_idx] = \
                np.clip(self.overall_event_counts[:, risk_idx] /
                        self.overall_at_risk_counts,
                        1e-12, 1. - 1e-12)
        self.overall_KM = \
            np.exp(
                np.cumsum(np.log(1 - self.overall_hazard.sum(axis=1) + 1e-7)))

        if self.beta == 0:
            ann_index = hnswlib.Index(space='l2',
                                      dim=train_embeddings.shape[1])
            if self.ann_deterministic_construction:
                ann_index.set_num_threads(1)
            else:
                ann_index.set_num_threads(os.cpu_count())
            ann_index.init_index(max_elements=n_train,
                                 ef_construction=2*self.n_neighbors,
                                 M=48,
                                 random_seed=self.ann_random_seed)
            ann_index.add_items(train_embeddings, np.arange(n_train))
            ann_index.set_ef(min(2*self.n_neighbors, n_train))

            self.ann_index = ann_index

            # store some helpful information for prediction
            train_event_array = np.zeros((n_train, num_durations, n_risks))
            train_obs_array = np.zeros((n_train, num_durations))
            for row, (idx, event) in enumerate(zip(idx_durations, events)):
                if event > 0:
                    train_event_array[row, idx, event - 1] = 1
                train_obs_array[row, idx] = 1
            self.train_event_array = train_event_array
            self.train_obs_array = train_obs_array

        else:
            eps = self.beta * self.tau
            eps_squared = eps * eps

            # build epsilon-net
            ann_index = \
                hnswlib.Index(space='l2', dim=train_embeddings.shape[1])
            ann_index.set_num_threads(1)
            ann_index.init_index(max_elements=n_train,
                                 ef_construction=2*self.dkn_n_neighbors,
                                 M=48,
                                 random_seed=self.ann_random_seed)
            ann_index.add_items([train_embeddings[0]], [0])
            assignments = [[0]]
            for idx in range(1, n_train):
                pt = train_embeddings[idx]
                labels, sq_dists = ann_index.knn_query([pt], k=1)
                nearest_exemplar_idx = labels[0, 0]
                sq_dist = sq_dists[0, 0]
                if sq_dist > eps_squared:
                    ann_index.add_items([pt], [len(assignments)])
                    assignments.append([idx])
                else:
                    assignments[nearest_exemplar_idx].append(idx)
            n_exemplars = len(assignments)
            if n_exemplars < self.dkn_n_neighbors:
                self.dkn_n_neighbors = n_exemplars
                if self.n_neighbors > self.dkn_n_neighbors:
                    self.n_neighbors = self.dkn_n_neighbors
            ann_index.set_ef(min(2*self.dkn_n_neighbors, n_exemplars))

            self.ann_index = ann_index

            # compute summary statistics
            self.exemplar_assignments = [np.array(_) for _ in assignments]
            self.exemplar_sizes = np.array([len(_) for _ in assignments])
            summaries = \
                [survival_summarize_cr(idx_durations[_], events[_],
                                       num_durations, n_risks)
                 for _ in assignments]
            self.exemplar_event_counts = np.array([x for x, _ in summaries])
            self.exemplar_at_risk_counts = np.array([x for _, x in summaries])

        self.baseline_event_counts = None
        self.baseline_at_risk_counts = None
        self.baseline_hazard = None

    def interpolate(self, sub=10, scheme='const_pdf', duration_index=None):
        return NotImplementedError

    def predict_cif(self, input, batch_size=256, numpy=None, eval_=True,
                    to_cpu=False, num_workers=0, epsilon=1e-7, mode='hazard',
                    multiply_by_surv_at_prev_time_step=True, **kwargs):
        event_specific_hazards, weighted_overall_hazards = \
            self.predict_hazard(input, batch_size, False, eval_,
                                to_cpu, num_workers, **kwargs)
        surv = \
            (1 - weighted_overall_hazards).add(epsilon).log().cumsum(1).exp()
        if multiply_by_surv_at_prev_time_step:
            surv_shifted = torch.roll(surv, shifts=1, dims=1)
            surv_shifted[:, 0] = 1.
        else:
            surv_shifted = surv
        cifs = torch.stack([(event_specific_hazards[risk_idx]
                             * surv_shifted).cumsum(1)
                            for risk_idx in range(self.n_risks)], dim=0)
        cifs = torch.transpose(cifs, 1, 2)
        return tt.utils.array_or_tensor(cifs, numpy, input)

    def predict_hazard(self, input, batch_size=256, numpy=None, eval_=True,
                       to_cpu=False, num_workers=0, **kwargs):
        test_embeddings = self.predict(input, batch_size, False, eval_, False,
                                       True, num_workers)
        train_embeddings = self.train_embeddings
        tau_squared = self.tau ** 2

        if self.beta == 0:
            # compute kernel matrix
            if self.brute_force:
                sq_dists = cdist(test_embeddings, train_embeddings,
                                 'sqeuclidean')
                weights = np.exp(-sq_dists) * (sq_dists <= tau_squared)

            else:
                weights_shape = ((test_embeddings.shape[0],
                                  self.train_embeddings.shape[0]))
                if num_workers <= 0:
                    n_threads = os.cpu_count()
                else:
                    n_threads = num_workers

                k = self.n_neighbors
                labels, sq_dists = \
                    self.ann_index.knn_query(test_embeddings, k=k,
                                             num_threads=n_threads)
                row_col_pairs = \
                    np.array([(row, col)
                              for row, cols in enumerate(labels)
                              for col in cols],
                             dtype=np.int64)
                rows = row_col_pairs[:, 0]
                cols = row_col_pairs[:, 1]
                sq_dists_flat = sq_dists.flatten()
                weights = csr_matrix((np.exp(-sq_dists_flat)
                                      * (sq_dists_flat <= tau_squared),
                                     (rows, cols)), shape=weights_shape)

            # convert observed time matrix to be for test data
            weights_discretized = weights.dot(self.train_obs_array)

            # kernel hazard function calculation
            num_at_risk = \
                np.flip(np.cumsum(np.flip(weights_discretized,
                                          axis=1), axis=1), axis=1) + 1e-12
            overall_numer = 0
            event_specific_hazards = []
            for risk_idx in range(self.n_risks):
                num_events = \
                    weights.dot(self.train_event_array[:, :, risk_idx])
                overall_numer += num_events

                hazards = np.zeros(num_at_risk.shape)
                row_sums = num_events.sum(axis=1)
                row_zero_mask = (row_sums == 0)
                if np.any(row_zero_mask):
                    hazards[row_zero_mask, :] = \
                        self.overall_hazard[:, risk_idx]
                row_nonzero_mask = ~row_zero_mask
                hazards[row_nonzero_mask] = \
                    np.clip(num_events[row_nonzero_mask]
                            / num_at_risk[row_nonzero_mask], 1e-12, 1. - 1e-12)
                event_specific_hazards.append(hazards)
            event_specific_hazards = np.array(event_specific_hazards)

            weighted_overall_hazards = np.zeros(num_at_risk.shape)
            row_sums = overall_numer.sum(axis=1)
            row_zero_mask = (row_sums == 0)
            if np.any(row_zero_mask):
                weighted_overall_hazards[row_zero_mask, :] = \
                    self.overall_hazard.sum(axis=1)
            row_nonzero_mask = ~row_zero_mask
            weighted_overall_hazards[row_nonzero_mask] = \
                np.clip(overall_numer[row_nonzero_mask]
                        / num_at_risk[row_nonzero_mask], 1e-12, 1. - 1e-12)

        else:
            if self.brute_force:
                exemplars = np.array([_[0] for _ in self.exemplar_assignments],
                                     dtype=np.int64)
                sq_dists = cdist(test_embeddings, train_embeddings[exemplars],
                                 'sqeuclidean')
                weights = np.exp(-sq_dists) * (sq_dists <= tau_squared)

            else:
                weights_shape = ((test_embeddings.shape[0],
                                  len(self.exemplar_assignments)))
                if num_workers <= 0:
                    n_threads = os.cpu_count()
                else:
                    n_threads = num_workers

                k = self.dkn_n_neighbors
                labels, sq_dists = \
                    self.ann_index.knn_query(test_embeddings, k=k,
                                             num_threads=n_threads)
                row_col_pairs = \
                    np.array([(row, col)
                              for row, cols in enumerate(labels)
                              for col in cols],
                             dtype=np.int64)
                rows = row_col_pairs[:, 0]
                cols = row_col_pairs[:, 1]
                sq_dists_flat = sq_dists.flatten()
                weights = csr_matrix((np.exp(-sq_dists_flat)
                                      * (sq_dists_flat <= tau_squared),
                                     (rows, cols)), shape=weights_shape)

            denom = weights.dot(self.exemplar_at_risk_counts)
            if self.baseline_at_risk_counts is not None:
                denom = denom + self.baseline_at_risk_counts[np.newaxis, :]
            denom_nonzero_mask = (denom > 0)
            any_denom_nonzero = np.any(denom_nonzero_mask)

            overall_numer = 0
            event_specific_hazards = []
            for risk_idx in range(self.n_risks):
                numer = weights.dot(self.exemplar_event_counts[:, :, risk_idx])
                hazards = np.zeros(numer.shape)

                if self.baseline_event_counts is not None:
                    numer = numer + \
                        self.baseline_event_counts[np.newaxis, :, risk_idx]

                overall_numer += numer

                if any_denom_nonzero:
                    hazards[denom_nonzero_mask] = numer[denom_nonzero_mask] / \
                        denom[denom_nonzero_mask]
                row_sums = numer.sum(axis=1)
                row_zero_mask = (row_sums == 0)
                if np.any(row_zero_mask):
                    hazards[row_zero_mask, :] = \
                        self.overall_hazard[:, risk_idx]

                hazards = np.clip(hazards, 1e-12, 1. - 1e-12)
                event_specific_hazards.append(hazards)

            weighted_overall_hazards = np.zeros(denom.shape)
            if any_denom_nonzero:
                weighted_overall_hazards[denom_nonzero_mask] = \
                    overall_numer[denom_nonzero_mask] / \
                    denom[denom_nonzero_mask]
            row_sums = overall_numer.sum(axis=1)
            row_zero_mask = (row_sums == 0)
            if np.any(row_zero_mask):
                weighted_overall_hazards[row_zero_mask, :] = \
                    self.overall_hazard.sum(axis=1)
            event_specific_hazards = np.array(event_specific_hazards)

            weighted_overall_hazards = np.clip(weighted_overall_hazards,
                                               1e-12, 1. - 1e-12)

        return \
            tt.utils.array_or_tensor(event_specific_hazards, numpy, input), \
            tt.utils.array_or_tensor(weighted_overall_hazards, numpy, input)

    def fit(self, input, target=None, batch_size=256, epochs=1, callbacks=None,
            verbose=True, num_workers=0, shuffle=True, metrics=None,
            val_data=None, val_batch_size=256, **kwargs):
        raise NotImplementedError

    def save_net(self, path, **kwargs):
        path, extension = os.path.splitext(path)
        assert extension == '.pt'
        super().save_model_weights(path + extension, **kwargs)

    def load_net(self, path, **kwargs):
        path, extension = os.path.splitext(path)
        assert extension == '.pt'
        super().load_model_weights(path + extension, **kwargs)

    def get_summary_functions(self):
        return self.exemplar_event_counts.copy(), \
            self.exemplar_at_risk_counts.copy()

    def load_summary_functions(self, exemplar_event_counts,
                               exemplar_at_risk_counts,
                               baseline_event_counts=None,
                               baseline_at_risk_counts=None):
        self.exemplar_event_counts = exemplar_event_counts
        self.exemplar_at_risk_counts = exemplar_at_risk_counts
        self.baseline_event_counts = baseline_event_counts
        self.baseline_at_risk_counts = baseline_at_risk_counts


def survival_summarize_cr(idx_durations, events, num_durations, n_risks,
                          weights=None):
    unnormalized_Z_q_bar = np.zeros((num_durations, n_risks))
    unnormalized_Z_q_plus = np.zeros(num_durations)
    if weights is None:
        for idx, event in zip(idx_durations, events):
            if event > 0:
                unnormalized_Z_q_bar[idx, event - 1] += 1
            unnormalized_Z_q_plus[idx] += 1
    else:
        for idx, event, weight in zip(idx_durations, events, weights):
            if event > 0:
                unnormalized_Z_q_bar[idx, event - 1] += weight
            unnormalized_Z_q_plus[idx] += weight
    unnormalized_Z_q_plus = np.flip(np.cumsum(np.flip(unnormalized_Z_q_plus)))
    return unnormalized_Z_q_bar, unnormalized_Z_q_plus


class DKAJSummaryLoss(nn.Module):
    def __init__(self, alpha, sigma):
        super(DKAJSummaryLoss, self).__init__()
        self.alpha = alpha
        self.sigma = sigma

    def forward(self, event_specific_hazards: Tensor,
                weighted_overall_hazards: Tensor,
                idx_durations: Tensor, events: Tensor,
                reduction: str = 'mean',
                multiply_by_surv_at_prev_time_step: bool = True) -> Tensor:
        if not torch.all((event_specific_hazards >= 0)
                         & (event_specific_hazards <= 1)):
            # weird corner case
            return torch.tensor(np.inf, dtype=torch.float32)

        if events.dtype is not torch.float:
            events = events.float()

        alpha = self.alpha
        sigma = self.sigma

        n_risks = event_specific_hazards.size(0)
        batch_size = event_specific_hazards.size(1)
        num_durations = event_specific_hazards.size(2)
        device = event_specific_hazards.device

        rank_matrices = []
        if alpha < 1:
            idx_durations_np = idx_durations.detach().cpu().numpy().ravel()
            events_np = events.detach().cpu().numpy().ravel()
            for risk_idx in range(n_risks):
                rank_mat_np = np.zeros((batch_size, batch_size),
                                       dtype='float32')
                rank_matrices.append(
                    torch.tensor(
                        _pair_rank_mat_cr(
                            rank_mat_np, idx_durations_np, events_np,
                            risk_idx + 1, 'float32'
                        ),
                        device=device
                    )
                )

        idx_durations = idx_durations.view(-1, 1)
        events = events.view(-1, 1) - 1

        obs_time_matrix = \
            torch.zeros((batch_size, num_durations),
                        dtype=torch.float,
                        device=device).scatter(1, idx_durations, 1)
        B = ((obs_time_matrix.flip(1)).cumsum(1)).flip(1)

        nll_loss = torch.zeros(batch_size, dtype=torch.float32, device=device)
        for risk_idx in range(n_risks):
            A = torch.zeros((batch_size, num_durations), dtype=torch.float,
                            device=device).scatter(1, idx_durations,
                1*(events == risk_idx).to(dtype=torch.float))

            nll_loss -= (A * torch.log(event_specific_hazards[risk_idx])
                         - B * event_specific_hazards[risk_idx]).sum(1)

        if alpha < 1:
            surv = \
                (1 - weighted_overall_hazards).add(1e-7).log().cumsum(1).exp()
            rank_loss = 0
            if multiply_by_surv_at_prev_time_step:
                surv_shifted = torch.roll(surv, shifts=1, dims=1)
                surv_shifted[:, 0] = 1.
            else:
                surv_shifted = surv
            for risk_idx in range(n_risks):
                cif = (event_specific_hazards[risk_idx] *
                       surv_shifted).cumsum(1)
                event_matrix = torch.zeros((batch_size, num_durations),
                                           dtype=torch.float,
                                           device=device).scatter(
                    1, idx_durations, 1.*(events == risk_idx))
                A = cif.matmul(event_matrix.transpose(0, 1))
                diag_A = A.diag().view(1, -1)
                ones = torch.ones((batch_size, 1), device=device)
                differences = (A - ones.matmul(diag_A)).transpose(0, 1)
                rank_loss += \
                    (rank_matrices[risk_idx] *
                     torch.exp(differences / sigma)).mean(1, keepdim=True)

            return alpha * pycox.models.loss._reduction(nll_loss, reduction) + \
                (1 - alpha) * pycox.models.loss._reduction(rank_loss, reduction)
        else:
            return pycox.models.loss._reduction(nll_loss, reduction)


class DKAJSummary(nn.Module):
    def __init__(self, kernel_model, exemplar_event_counts,
                 exemplar_at_risk_counts):
        super(DKAJSummary, self).__init__()
        self.ann_index = kernel_model.ann_index
        self.dkn_n_neighbors = kernel_model.dkn_n_neighbors
        self.tau_squared = kernel_model.tau**2
        device = kernel_model.device
        exemplar_indices = \
            np.array([_[0] for _ in kernel_model.exemplar_assignments],
                     dtype=np.int64)
        exemplar_embeddings = kernel_model.train_embeddings[exemplar_indices]
        if type(exemplar_embeddings) is not torch.Tensor:
            exemplar_embeddings = torch.tensor(exemplar_embeddings,
                                               dtype=torch.float32,
                                               device=device)
        else:
            exemplar_embeddings = torch.clone(exemplar_embeddings.detach())
        self.exemplar_embeddings = exemplar_embeddings
        assert type(exemplar_event_counts) == np.ndarray
        assert type(exemplar_at_risk_counts) == np.ndarray
        exemplar_at_risk_counts = \
            np.hstack((exemplar_at_risk_counts,
                       np.zeros((exemplar_at_risk_counts.shape[0], 1))))
        exemplar_censor_counts = \
            exemplar_at_risk_counts[:, :-1] - exemplar_at_risk_counts[:, 1:] \
            - exemplar_event_counts.sum(axis=2)
        self.log_exemplar_event_counts = \
            nn.Parameter(torch.tensor(exemplar_event_counts,
                                      dtype=torch.float32,
                                      device=device).clamp(
                                          min=1e-12).log())
        self.log_exemplar_censor_counts = \
            nn.Parameter(torch.tensor(exemplar_censor_counts,
                                      dtype=torch.float32,
                                      device=device).clamp(
                                          min=1e-12).log())
        n_durations = exemplar_event_counts.shape[1]
        n_risks = exemplar_event_counts.shape[2]
        self.log_baseline_event_counts = \
            nn.Parameter(-27.6310211159 * torch.ones(n_durations, n_risks,
                                                     dtype=torch.float32,
                                                     device=device))
        self.log_baseline_censor_counts = \
            nn.Parameter(-27.6310211159 * torch.ones(n_durations,
                                                     dtype=torch.float32,
                                                     device=device))

    def forward(self, input, num_workers: int = 0) -> Tensor:
        # --------------------------------------------------------------------
        # Get kernel weights (done on CPU; at this point the embedding space
        # and thus also the kernel weights are treated as fixed)
        #
        if type(input) == torch.Tensor:
            input_np = input.detach().cpu().numpy()
        else:
            input_np = input

        weights_shape = (input_np.shape[0], self.exemplar_embeddings.size(0))
        if num_workers <= 0:
            n_threads = os.cpu_count()
        else:
            n_threads = num_workers
        labels, sq_dists = \
            self.ann_index.knn_query(input_np,
                                     k=self.dkn_n_neighbors,
                                     num_threads=n_threads)
        row_col_pairs = \
            np.array([(row, col)
                      for row, cols in enumerate(labels)
                      for col in cols],
                     dtype=np.int64)
        rows = row_col_pairs[:, 0]
        cols = row_col_pairs[:, 1]
        sq_dists_flat = sq_dists.flatten()
        weights = csr_matrix((np.exp(-sq_dists_flat)
                              * (sq_dists_flat <= self.tau_squared),
                             (rows, cols)), shape=weights_shape)
        kernel_weights = \
            torch.tensor(weights.toarray(), dtype=torch.float32,
                         device=self.log_baseline_event_counts.device)

        # --------------------------------------------------------------------
        # Compute hazard functions
        #
        baseline_event_counts = self.log_baseline_event_counts.exp()
        baseline_censor_counts = self.log_baseline_censor_counts.exp()
        baseline_at_risk_counts = \
            (baseline_event_counts.sum(1)
             + baseline_censor_counts).flip(0).cumsum(0).flip(0)

        exemplar_event_counts = self.log_exemplar_event_counts.exp()
        exemplar_censor_counts = self.log_exemplar_censor_counts.exp()
        exemplar_at_risk_counts = \
            (exemplar_event_counts.sum(2)
             + exemplar_censor_counts).flip(1).cumsum(1).flip(1)

        denom = torch.matmul(kernel_weights, exemplar_at_risk_counts) \
            + baseline_at_risk_counts.view(1, -1) + 1e-12
        overall_numer = 0
        event_specific_hazards = []
        for risk_idx in range(exemplar_event_counts.size(2)):
            numer = torch.matmul(kernel_weights,
                                 exemplar_event_counts[:, :, risk_idx]) \
                + baseline_event_counts[:, risk_idx].view(1, -1)
            overall_numer += numer
            event_specific_hazards.append(
                torch.clamp(numer / denom, 1e-12, 1. - 1e-12)
            )
        event_specific_hazards = torch.stack(event_specific_hazards)
        weighted_overall_hazards = \
            torch.clamp(overall_numer / denom, 1e-12, 1. - 1e-12)

        return event_specific_hazards, weighted_overall_hazards

    def get_exemplar_summary_functions_baseline_event_at_risk_counts(self):
        log_exemplar_event_counts = self.log_exemplar_event_counts
        log_exemplar_censor_counts = self.log_exemplar_censor_counts
        log_baseline_event_counts = self.log_baseline_event_counts
        log_baseline_censor_counts = self.log_baseline_censor_counts

        exemplar_event_counts = \
            np.exp(log_exemplar_event_counts.detach(
                ).cpu().numpy())
        exemplar_censor_counts = \
            np.exp(log_exemplar_censor_counts.detach(
                ).cpu().numpy())
        exemplar_at_risk_counts = \
            np.flip(
                np.cumsum(
                    np.flip(exemplar_event_counts.sum(axis=2)
                            + exemplar_censor_counts,
                            axis=1),
                    axis=1),
                axis=1)
        baseline_event_counts = \
            np.exp(log_baseline_event_counts.detach(
                ).cpu().numpy())
        baseline_censor_counts = \
            np.exp(log_baseline_censor_counts.detach(
                ).cpu().numpy())
        baseline_at_risk_counts = \
            np.flip(
                np.cumsum(
                    np.flip(baseline_event_counts.sum(axis=1)
                            + baseline_censor_counts)))

        n_exemplars = exemplar_event_counts.shape[0]
        n_time = exemplar_event_counts.shape[1]
        return exemplar_event_counts, exemplar_at_risk_counts, \
            baseline_event_counts, baseline_at_risk_counts
