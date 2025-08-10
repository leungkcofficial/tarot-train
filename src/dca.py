import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

def risk_at_horizon(surv_data, horizon):
    """
    Convert survival probabilities to risk at a specific time horizon.
    
    Args:
        surv_data: DataFrame containing survival probabilities (rows=time points, columns=patients)
                   OR numpy array for DeepHit competing risks (shape: causes, time_points, samples)
        horizon: Time horizon in days
        
    Returns:
        Array of risk values (1 - survival probability) for each patient at the specified horizon
    """
    import pandas as pd
    import numpy as np
    
    # Handle pandas DataFrame (DeepSurv format)
    if isinstance(surv_data, pd.DataFrame):
        idx = surv_data.index.get_indexer([horizon], method="nearest")[0]
        return 1.0 - surv_data.iloc[idx].values  # ndarray (n_patients,)
    
    # Handle numpy array (DeepHit competing risks format)
    elif isinstance(surv_data, np.ndarray):
        if len(surv_data.shape) == 3:  # (causes, time_points, samples)
            # For competing risks, use the first cause (Event 1: RRT/eGFR<15)
            # Convert from CIF to survival probability: survival = 1 - CIF
            # Then convert to risk: risk = 1 - survival = CIF
            cause_1_cif = surv_data[0]  # Shape: (time_points, samples)
            
            # Find the closest time point to the horizon
            # Assume time points are [365, 730, 1095, 1460, 1825] (years 1-5)
            time_points = np.array([365, 730, 1095, 1460, 1825])
            idx = np.argmin(np.abs(time_points - horizon))
            
            # Return CIF values (which represent risk for competing events)
            return cause_1_cif[idx]  # ndarray (n_samples,)
        elif len(surv_data.shape) == 2:  # (time_points, samples) - single cause
            # Handle 2D array case for single cause from event-specific evaluation
            # Find the closest time point to the horizon
            time_points = np.array([365, 730, 1095, 1460, 1825])
            idx = np.argmin(np.abs(time_points - horizon))
            
            # Return CIF values (which represent risk for competing events)
            return surv_data[idx]  # ndarray (n_samples,)
        else:
            # Handle other cases
            raise ValueError(f"Unexpected array shape for DeepHit predictions: {surv_data.shape}")
    
    else:
        raise TypeError(f"Unsupported data type for surv_data: {type(surv_data)}")

def ipcw_weights(durations, events, horizon):
    """
    Calculate inverse probability of censoring weights.
    
    Args:
        durations: Array of event/censoring times
        events: Array of event indicators (1=event, 0=censored)
        horizon: Time horizon in days
        
    Returns:
        Array of weights for each patient
    """
    km = KaplanMeierFitter().fit(durations, event_observed=(events == 0))
    G_t = km.survival_function_at_times(horizon).values[0]
    G_i = km.predict(np.minimum(durations, horizon))
    return 1.0 / np.where(durations > horizon, G_t, G_i)

def decision_curve(risk, durations, events, horizon, thresholds=None, ipcw=True):
    """
    Calculate net benefit for different risk thresholds.
    
    Args:
        risk: Array of predicted risks at the specified horizon
        durations: Array of event/censoring times
        events: Array of event indicators (1=event, 0=censored)
        horizon: Time horizon in days
        thresholds: Array of risk thresholds (default: np.linspace(0.01, 0.5, 50))
        ipcw: Whether to use inverse probability of censoring weighting (default: True)
        
    Returns:
        Dictionary containing thresholds, net benefit for the model, treat all, and treat none
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.5, 50)
    
    w = ipcw_weights(durations, events, horizon) if ipcw else np.ones_like(risk)
    event = (events == 1) & (durations <= horizon)

    def nb(thr):
        treated = risk >= thr
        tp = np.sum(w * (treated & event))
        fp = np.sum(w * (treated & ~event))
        N = np.sum(w)
        exchange = thr / (1 - thr)
        return (tp / N) - (fp / N) * exchange

    nb_model = np.array([nb(t) for t in thresholds])
    preval = np.average(event, weights=w)
    nb_all = preval - (1 - preval) * (thresholds / (1 - thresholds))
    nb_none = np.zeros_like(thresholds)

    return dict(thresholds=thresholds, nb_model=nb_model,
                nb_treat_all=nb_all, nb_treat_none=nb_none)

def plot_decision_curve(thresholds, nb_model, nb_all, nb_none, label, out_path):
    """
    Create and save a decision curve plot.
    
    Args:
        thresholds: Array of risk thresholds
        nb_model: Array of net benefit values for the model
        nb_all: Array of net benefit values for treating all patients
        nb_none: Array of net benefit values for treating no patients
        label: Label for the model curve
        out_path: Path to save the plot
        
    Returns:
        Path to the saved plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, nb_model, '-', linewidth=2, label=label)
    plt.plot(thresholds, nb_all, '--', linewidth=1.5, label="Treat All")
    plt.plot(thresholds, nb_none, '-', linewidth=1.5, label="Treat None")
    plt.xlabel("Risk Threshold")
    plt.ylabel("Net Benefit (%)")
    plt.title("Decision Curve Analysis")
    plt.legend(loc="best")
    plt.grid(True, linestyle='--', alpha=0.7)
    # Set y-axis lower limit to 0 to hide negative net benefit values
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path

def plot_decision_curves_subplots(dca_data_list, out_path):
    """
    Create and save a decision curve plot with multiple subplots.
    
    Args:
        dca_data_list: List of dictionaries, each containing:
            - thresholds: Array of risk thresholds
            - nb_model: Array of net benefit values for the model
            - nb_all: Array of net benefit values for treating all patients
            - nb_none: Array of net benefit values for treating no patients
            - label: Label for the model curve
            - horizon: Time horizon for the subplot title
        out_path: Path to save the plot
        
    Returns:
        Path to the saved plot
    """
    n_plots = len(dca_data_list)
    
    # Create subplots in a single row
    fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 6))
    
    # Handle case where there's only one subplot
    if n_plots == 1:
        axes = [axes]
    
    for i, dca_data in enumerate(dca_data_list):
        ax = axes[i]
        
        # Plot the curves
        ax.plot(dca_data['thresholds'], dca_data['net_benefit'], '-',
                linewidth=2, label=dca_data['label'])
        ax.plot(dca_data['thresholds'], dca_data['net_benefit_all'], '--',
                linewidth=1.5, label="Treat All")
        ax.plot(dca_data['thresholds'], dca_data['net_benefit_none'], '-',
                linewidth=1.5, label="Treat None")
        
        # Set labels and title
        ax.set_xlabel("Risk Threshold")
        ax.set_ylabel("Net Benefit (%)")
        ax.set_title(f"DCA - {dca_data['horizon']} days")
        ax.legend(loc="best")
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set y-axis lower limit to 0 to hide negative net benefit values
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path