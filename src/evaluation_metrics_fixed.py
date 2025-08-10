"""
Evaluation metrics for survival analysis with competing risks

Implements:
- Integrated Brier Score (IBS)
- Concordance Index (C-index)
- Negative Log-Likelihood (NLL)
"""

import numpy as np
from typing import Tuple, Optional
from sksurv.metrics import concordance_index_censored as sklearn_concordance_index_censored
try:
    from scipy.integrate import trapz
except ImportError:
    # In newer scipy versions, trapz is in numpy
    from numpy import trapz


def integrated_brier_score(times: np.ndarray,
                         events: np.ndarray,
                         predictions: np.ndarray,
                         time_points: np.ndarray,
                         event_of_interest: Optional[int] = None) -> float:
    """
    Calculate Integrated Brier Score for competing risks
    
    Args:
        times: Observed times (n_samples,)
        events: Observed events (n_samples,) with values 0 (censored), 1, 2
        predictions: CIF predictions of shape (n_events, n_time_points, n_samples)
        time_points: Time points at which predictions are made
        event_of_interest: If specified, calculate IBS for specific event only
        
    Returns:
        IBS score (lower is better)
    """
    n_events, n_time_points, n_samples = predictions.shape
    
    # Initialize Brier scores
    brier_scores = []
    
    for t_idx, t in enumerate(time_points):
        bs_t = 0
        
        if event_of_interest is not None:
            # Single event IBS
            event_list = [event_of_interest]
        else:
            # All events
            event_list = range(1, n_events + 1)
        
        for event in event_list:
            # Get predictions for this event at time t
            pred_event_t = predictions[event - 1, t_idx, :]
            
            # Calculate weights (inverse probability of censoring weights)
            # For simplicity, using uniform weights here
            # In practice, you might want to use Kaplan-Meier estimates
            weights = np.ones(n_samples)
            
            # Observed indicator: did event occur before time t?
            observed = (times <= t) & (events == event)
            
            # Brier score component
            bs_component = weights * (observed - pred_event_t) ** 2
            bs_t += np.mean(bs_component)
        
        brier_scores.append(bs_t)
    
    # Integrate over time using trapezoidal rule
    ibs = trapz(brier_scores, time_points) / (time_points[-1] - time_points[0])
    
    return ibs


def concordance_index_censored(event_indicator: np.ndarray,
                             event_time: np.ndarray,
                             estimate: np.ndarray) -> Tuple[float, int, int, int, int]:
    """
    Wrapper for scikit-survival's concordance_index_censored
    
    Args:
        event_indicator: Boolean array indicating if event occurred
        event_time: Time of event or censoring
        estimate: Risk scores (higher values indicate higher risk)
        
    Returns:
        Tuple of (c-index, concordant, discordant, tied_risk, tied_time)
    """
    return sklearn_concordance_index_censored(event_indicator, event_time, estimate)


def negative_log_likelihood(times: np.ndarray,
                          events: np.ndarray,
                          predictions: np.ndarray,
                          time_points: np.ndarray,
                          epsilon: float = 1e-7) -> float:
    """
    Calculate negative log-likelihood for competing risks
    
    Args:
        times: Observed times (n_samples,)
        events: Observed events (n_samples,) with values 0 (censored), 1, 2
        predictions: CIF predictions of shape (n_events, n_time_points, n_samples)
        time_points: Time points at which predictions are made
        epsilon: Small value to avoid log(0)
        
    Returns:
        NLL score (lower is better)
    """
    n_events, n_time_points, n_samples = predictions.shape
    
    nll = 0
    
    for i in range(n_samples):
        t_i = times[i]
        e_i = int(events[i])  # Ensure integer for indexing
        
        if e_i == 0:  # Censored
            # For censored observations, we need the survival probability
            # S(t) = 1 - sum of all CIFs at time t
            
            # Find the time point closest to observed time
            t_idx = np.searchsorted(time_points, t_i)
            if t_idx >= n_time_points:
                t_idx = n_time_points - 1
            
            # Calculate survival probability
            total_cif = np.sum(predictions[:, t_idx, i])
            surv_prob = 1 - total_cif
            surv_prob = np.clip(surv_prob, epsilon, 1 - epsilon)
            
            nll -= np.log(surv_prob)
            
        else:  # Event occurred
            # For events, we need the probability density
            # This is approximated by the difference in CIF
            
            # Find the time interval containing the event
            t_idx = np.searchsorted(time_points, t_i)
            
            if t_idx == 0:
                # Event before first time point
                cif_prob = predictions[e_i - 1, 0, i]
            elif t_idx >= n_time_points:
                # Event after last time point
                cif_prob = predictions[e_i - 1, -1, i] - predictions[e_i - 1, -2, i]
            else:
                # Event between time points
                # Interpolate to get more accurate probability
                t_prev = time_points[t_idx - 1]
                t_next = time_points[t_idx] if t_idx < n_time_points else time_points[-1]
                
                cif_prev = predictions[e_i - 1, t_idx - 1, i]
                cif_next = predictions[e_i - 1, min(t_idx, n_time_points - 1), i]
                
                # Linear interpolation
                alpha = (t_i - t_prev) / (t_next - t_prev) if t_next > t_prev else 0
                cif_at_t = cif_prev + alpha * (cif_next - cif_prev)
                
                # Approximate density
                if t_idx > 0:
                    cif_prob = cif_at_t - cif_prev
                else:
                    cif_prob = cif_at_t
            
            cif_prob = np.clip(cif_prob, epsilon, 1 - epsilon)
            nll -= np.log(cif_prob)
    
    return nll / n_samples


def calculate_all_metrics(times: np.ndarray,
                        events: np.ndarray,
                        predictions: np.ndarray,
                        time_points: np.ndarray) -> dict:
    """
    Calculate all evaluation metrics
    
    Args:
        times: Observed times
        events: Observed events (0, 1, or 2)
        predictions: CIF predictions (n_events, n_time_points, n_samples)
        time_points: Time points for predictions
        
    Returns:
        Dictionary with all metrics
    """
    results = {}
    
    # Integrated Brier Score
    try:
        results['ibs'] = integrated_brier_score(times, events, predictions, time_points)
        results['ibs_event1'] = integrated_brier_score(times, events, predictions, time_points, event_of_interest=1)
        results['ibs_event2'] = integrated_brier_score(times, events, predictions, time_points, event_of_interest=2)
    except Exception as e:
        print(f"IBS calculation failed: {e}")
        results['ibs'] = np.nan
        results['ibs_event1'] = np.nan
        results['ibs_event2'] = np.nan
    
    # Concordance Index for Event 1
    event1_mask = events != 2
    if np.sum(event1_mask) > 10:
        try:
            # Get predictions for event 1 at last time point
            event1_predictions = predictions[0, -1, :]
            results['cidx_event1'] = concordance_index_censored(
                events[event1_mask] == 1,
                times[event1_mask],
                event1_predictions[event1_mask]  # Positive because higher CIF = higher risk
            )[0]
        except Exception as e:
            print(f"C-index Event 1 calculation failed: {e}")
            results['cidx_event1'] = np.nan
    else:
        results['cidx_event1'] = np.nan
    
    # Concordance Index for Event 2
    event2_mask = events != 1
    if np.sum(event2_mask) > 10:
        try:
            # Get predictions for event 2 at last time point
            event2_predictions = predictions[1, -1, :]
            results['cidx_event2'] = concordance_index_censored(
                events[event2_mask] == 2,
                times[event2_mask],
                event2_predictions[event2_mask]  # Positive because higher CIF = higher risk
            )[0]
        except Exception as e:
            print(f"C-index Event 2 calculation failed: {e}")
            results['cidx_event2'] = np.nan
    else:
        results['cidx_event2'] = np.nan
    
    # Negative Log-Likelihood
    try:
        results['nll'] = negative_log_likelihood(times, events, predictions, time_points)
    except Exception as e:
        print(f"NLL calculation failed: {e}")
        results['nll'] = np.nan
    
    return results