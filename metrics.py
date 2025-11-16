import numpy as np
from scipy.integrate import trapezoid
from pycox.utils import idx_at_times
from pycox.evaluation.concordance import concordance_td

def neg_cindex_td(y_true, d_true, surv_pred, exact=True, n_subsamples=None,
                  approx_batch_size=2**14):
    surv, times = surv_pred
    if exact:
        return -concordance_td(y_true, d_true, surv,
                               idx_at_times(times, y_true, 'post'),
                               'antolini')

    # note that for small datasets, no subsampling will actually be done by the
    # code below, so the calculation will still be exact
    n_samples = y_true.shape[0]
    if n_subsamples is not None:
        batch_size = int(np.ceil(n_samples / n_subsamples))
    else:
        if n_samples < approx_batch_size:
            batch_size = n_samples
        else:
            n_subsamples = int(np.ceil(n_samples / approx_batch_size))
            batch_size = int(np.ceil(n_samples / n_subsamples))

    c_indices = []
    for batch_start_idx in range(0, n_samples, batch_size):
        batch_end_idx = min(batch_start_idx + batch_size, n_samples)
        c_indices.append(
            concordance_td(y_true[batch_start_idx:batch_end_idx],
                           d_true[batch_start_idx:batch_end_idx],
                           surv[:, batch_start_idx:batch_end_idx],
                           idx_at_times(
                               times,
                               y_true[batch_start_idx:batch_end_idx],
                               'post'),
                           'antolini'))
    return -np.mean(c_indices)


# Section 4.1  Performance metrics in "Random survival forests for competing risks"
# Hemant Ishwaran, Thomas A. Gerds, Udaya B. Kogalur, Richard D. Moore, Stephen J. Gange, Bryan M. Lau, 
def c_index_competing_single_time(event_times, predicted_scores, event_observed, event_of_interest=1):
    '''
    Compute the concordance index for competing risks.
    Args:
        event_times: (n,) array of event times
        predicted_scores: (n, k) array of predicted scores
        event_observed: (n,) array of event observed indicator
        event_of_interest: int, the event of interest
    '''
    raise NotImplementedError(f"Not implemented yet")

def compute_brier_competing_single_time(cif_values_at_time_horizon, censoring_kmf,
                            Y_test, D_test, event_of_interest, time_horizon):
    '''
    Return Brier score at a single time horizon under competing risks setting.
    '''
    n = len(Y_test)
    assert len(D_test) == n

    # Precompute weights
    w_time = censoring_kmf.predict(time_horizon)
    w_obs  = censoring_kmf.predict(Y_test)  # vector of weights for each observed time

    # Case masks
    mask_later = Y_test > time_horizon
    mask_event = (Y_test <= time_horizon) & (D_test == event_of_interest)
    mask_comp  = (Y_test <= time_horizon) & ((D_test != event_of_interest) & (D_test != 0))

    # Residuals
    residuals = np.zeros_like(Y_test, dtype=float)
    residuals[mask_later] = (cif_values_at_time_horizon[mask_later] ** 2) / w_time
    residuals[mask_event] = ((1 - cif_values_at_time_horizon[mask_event]) ** 2) / w_obs[mask_event]
    residuals[mask_comp]  = (cif_values_at_time_horizon[mask_comp] ** 2) / w_obs[mask_comp]

    return residuals.mean()


def compute_brier_competing_multiple_times(cif_values_grid, censoring_kmf,
                                           Y_test, D_test, event_of_interest,
                                           time_horizons):
    '''
    Return Brier scores at multiple time horizons under competing risks setting.

    Parameters
    ----------
    cif_values_grid: array-like, shape (n_samples, n_timepoints)
        Cumulative incidence function values of the `event_of_interest` for each sample at each time point.
    
    censoring_kmf: fitted lifelines.KaplanMeierFitter for the censoring distribution.
    
    event_of_interest: int
        The event of interest (e.g., 1 for the event 1, 2 for the event 2, 0 for censored).
    
    time_horizons: array-like, shape (n_timepoints,)
        The time points at which to evaluate the Brier score, corresponding to time horizons where the `cif_values` are evaluated.

    
    Returns
    -------
    brier_scores: array-like, shape (n_timepoints,)
        The Brier scores at each time point under competing risks setting.
    '''
    n_samples, n_timepoints = cif_values_grid.shape
    time_horizons = np.asarray(time_horizons)

    # Precompute weights
    w_horizons = censoring_kmf.predict(time_horizons).values   # (n_timepoints,)
    w_obs      = censoring_kmf.predict(Y_test).values          # (n_samples,)

    # Broadcast shapes
    cif = cif_values_grid                               # (n_samples, n_timepoints)
    Y   = Y_test[:, None]                               # (n_samples, 1)
    D   = D_test[:, None]                               # (n_samples, 1)
    T   = time_horizons[None, :]                        # (1, n_timepoints)

    # Full weight matrices
    Wt_full  = np.tile(w_horizons, (n_samples, 1))      # (n_samples, n_timepoints)
    Wobs_full = np.tile(w_obs[:, None], (1, n_timepoints))

    # Masks
    mask_later = Y > T
    mask_event = (Y <= T) & (D == event_of_interest)
    mask_comp  = (Y <= T) & ((D != event_of_interest) & (D != 0))

    # Residuals
    residuals = np.zeros_like(cif, dtype=float)
    residuals[mask_later] = (cif[mask_later] ** 2) / Wt_full[mask_later]
    residuals[mask_event] = ((1 - cif)[mask_event] ** 2) / Wobs_full[mask_event]
    residuals[mask_comp]  = (cif[mask_comp] ** 2) / Wobs_full[mask_comp]

    return residuals.mean(axis=0)   # (n_timepoints,)


def compute_ibs_competing(cif_values_grid, censoring_kmf, 
                          Y_test, D_test, event_of_interest, time_horizons):
    '''
    Compute Integrated Brier Score (IBS) under competing risks setting.
    Step 1: Compute Brier scores at multiple time horizons.
    Step 2: Integrate using trapezoidal rule.

    Parameters
    ----------
    cif_values_grid: array-like, shape (n_samples, n_timepoints)
        Cumulative incidence function values of the `event_of_interest` for each sample at each time point.
    
    censoring_kmf: fitted lifelines.KaplanMeierFitter for the censoring distribution.
    
    event_of_interest: int
        The event of interest (e.g., 1 for the event 1, 2 for the event 2, 0 for censored).
    
    time_horizons: array-like, shape (n_timepoints,)
        The time points at which to evaluate the Brier score, corresponding to time horizons where the `cif_values` are evaluated.

    
    Returns
    -------
    brier_score: float
        The Integrated Brier Score (IBS) under competing risks setting.
    '''
    n_samples = len(Y_test)
    n_timepoints = len(time_horizons)
    assert cif_values_grid.shape == (n_samples, n_timepoints)
    time_horizons = np.asarray(time_horizons)

    brier_scores = compute_brier_competing_multiple_times(
        cif_values_grid=cif_values_grid,
        censoring_kmf=censoring_kmf,
        Y_test=Y_test,
        D_test=D_test,
        event_of_interest=event_of_interest,
        time_horizons=time_horizons
    )
    return trapezoid(brier_scores, time_horizons) / (time_horizons[-1] - time_horizons[0])