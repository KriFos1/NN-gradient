import torch
from NeuralSim.image_to_log import EMProxy

import os
# fix the seed for reproducability
import numpy as np

# Set up device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
from geostat.gaussian_sim import fast_gaussian
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for parallel processing
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(10)

# remove folders in current directory starting with 'PF_'
for folder in os.listdir('.'):
    if folder.startswith('PF_') and os.path.isdir(folder):
        import shutil
        shutil.rmtree(folder)

# Particle filter parameters
ne = 10000  # number of particles
resample_threshold = 0.5  # effective sample size threshold for resampling

data = pd.read_pickle('data.pkl')
Cd = pd.read_pickle('var.pkl')
data_keys = list(Cd.columns)  # Extract column headers from Cd
datatype = [str(dat) for dat in data_keys]  # make into strings
param_keys = ['rh']

proxi_scalers = "https://gitlab.norceresearch.no/saly/image_to_log_weights/-/raw/master/em/{}.pth?ref_type=heads"
proxi_save_file = "https://gitlab.norceresearch.no/saly/image_to_log_weights/-/raw/master/em/checkpoint_770.pth?ref_type=heads"

nn_input_dict = {'input_shape': (3, 128),  # These are fixed
                 'output_shape': (6, 18),
                 'checkpoint_path': proxi_save_file,
                 'scaler': proxi_scalers
                 }
NN_sim = EMProxy(**nn_input_dict)
NN_sim = NN_sim.to(device)  # Move model to GPU

# read trajectory of well to ensure that the grid is aligned
T = np.loadtxt('../data/Benchmark-3/ascii/trajectory.DAT', comments='%')
TVD = T[:, 1]

# The grid does not have any depth. The well position is controlled by the one-hot vector, and the assumption that each grid cell is 0.5 m (1.64042 ft) thick.
# Given the well trajectory, the grid cell size, and the initial value of the one-hot vector. We can update the one-hot vector to match the well position at each measurement point.
# One strict assumption is that tvd[-1] - tvd[0] < 128 * 1.64042 ft
# Assert this
Dh = 1.64042  # ft. (0.5 m)
assert TVD[-1] - TVD[0] < 128 * Dh, "The well trajectory exceeds the grid depth range."

# Cell-center values
grid_size = 128
# Initial position in one-hot vector
well_start_index = 25  # starting at cell 25

# Cell center positions: TVD[0] is at center of cell 25
# Cell indices go from 0 to 127, with cell well_start_index at TVD[0]
cell_center_tvd = TVD[0] + (np.arange(grid_size) - well_start_index) * Dh

NN_data_names = [
    'real(xx)', 'img(xx)',
    'real(yy)', 'img(yy)',
    'real(zz)', 'img(zz)',
    'real(xz)', 'img(xz)',
    'real(zx)', 'img(zx)',
    'USDA', 'USDP',
    'UADA', 'UADP',
    'UHRA', 'UHRP',
    'UHAA', 'UHAP'
]
observed_data_order_udar = [
    'USDP', 'USDA',
    'UADP', 'UADA',
    'UHRP', 'UHRA',
    'UHAP', 'UHAA'
]

observed_data_order_bfield = ['real(xx)', 'real(xy)', 'real(xz)', 'real(yx)', 'real(yy)', 'real(yz)', 'real(zx)',
                              'real(zy)', 'real(zz)',
                              'img(xx)', 'img(xy)', 'img(xz)', 'img(yx)', 'img(yy)', 'img(yz)', 'img(zx)', 'img(zy)',
                              'img(zz)']

nn_to_obs_udar_mapping = {data_name: (10 + nn_index, observed_data_order_udar.index(data_name)) for nn_index, data_name
                           in enumerate(NN_data_names[10:])}
nn_to_obs_bfield_mapping = {data_name: (nn_index, observed_data_order_bfield.index(data_name)) for nn_index, data_name
                             in enumerate(NN_data_names[:10])}

data_type = 'UDAR'  # 'UDAR' or 'Bfield'
# mapping = nn_to_obs_bfield_mapping
mapping = nn_to_obs_udar_mapping

tool_info = [('6kHz','83ft'),('12kHz','83ft'),('24kHz','83ft'),('24kHz','43ft'),('48kHz','43ft'),('96kHz','43ft')]



def simulate_ensemble(param_ensemble, well_pos_index):
    """
    Simulate an ensemble of resistivity models in parallel.
    
    Parameters:
    -----------
    param_ensemble : np.ndarray, shape (ne, grid_size)
        Ensemble of resistivity values
    well_pos_index : int
        Index for well position in one-hot encoding
    
    Returns:
    --------
    predictions : np.ndarray, shape (ne, n_data)
        Predictions for each ensemble member (flattened)
    predictions_by_tool : np.ndarray, shape (ne, n_tools, n_measurements)
        Predictions organized by tool
    """
    ne = param_ensemble.shape[0]
    
    # Build resistivity tensor for entire ensemble: shape (ne, 3, grid_size)
    resistivity = np.zeros((ne, 3, grid_size))
    resistivity[:, 0, :] = param_ensemble  # rh
    resistivity[:, 1, :] = param_ensemble  # rv (same as rh)
    resistivity[:, 2, well_pos_index] = 1.0  # well position one-hot
    
    # Convert to tensor and run forward pass (batched)
    resistivity_tensor = torch.tensor(resistivity, dtype=torch.float32, device=device)
    nn_pred_tensor = NN_sim.image_to_log(resistivity_tensor)
    
    # Extract predictions: shape (ne, n_tools, n_measurements)
    nn_pred = nn_pred_tensor.detach().cpu().numpy()[:, :, [val[0] for val in mapping.values()]]
    
    # Reshape to (ne, n_data)
    predictions = nn_pred.reshape(ne, -1)
    
    return predictions, nn_pred


def calculate_log_likelihood(pred_ensemble, data_real, Cd):
    """
    Calculate log-likelihood for each particle.
    
    Parameters:
    -----------
    pred_ensemble : np.ndarray, shape (ne, n_data)
        Predictions for each particle
    data_real : np.ndarray, shape (n_data,)
        Observed data
    Cd : np.ndarray, shape (n_data,)
        Data variance
    
    Returns:
    --------
    log_likelihood : np.ndarray, shape (ne,)
        Log-likelihood for each particle
    """
    ne = pred_ensemble.shape[0]
    log_likelihood = np.zeros(ne)
    
    for i in range(ne):
        # Data misfit term - Gaussian likelihood
        residuals = data_real - pred_ensemble[i]
        log_likelihood[i] = -0.5 * np.sum((residuals ** 2) / Cd + np.log(2 * np.pi * Cd))
    
    return log_likelihood


def systematic_resampling(weights):
    """
    Systematic resampling of particles.
    
    Parameters:
    -----------
    weights : np.ndarray, shape (ne,)
        Normalized weights for each particle
    
    Returns:
    --------
    indices : np.ndarray, shape (ne,)
        Indices of resampled particles
    """
    ne = len(weights)
    positions = (np.arange(ne) + np.random.uniform(0, 1)) / ne
    
    cumulative_sum = np.cumsum(weights)
    indices = np.zeros(ne, dtype=int)
    
    i, j = 0, 0
    while i < ne:
        if positions[i] < cumulative_sum[j]:
            indices[i] = j
            i += 1
        else:
            j += 1
    
    return indices


def calculate_effective_sample_size(weights):
    """
    Calculate effective sample size.
    
    Parameters:
    -----------
    weights : np.ndarray, shape (ne,)
        Normalized weights
    
    Returns:
    --------
    ess : float
        Effective sample size
    """
    return 1.0 / np.sum(weights ** 2)


def add_jitter(ensemble, jitter_scale=0.01):
    """
    Add small random noise to ensemble to maintain diversity after resampling.
    
    Parameters:
    -----------
    ensemble : np.ndarray, shape (grid_size, ne)
        Current ensemble
    jitter_scale : float
        Scale of jitter relative to ensemble standard deviation
    
    Returns:
    --------
    jittered_ensemble : np.ndarray
        Ensemble with added jitter
    """
    std_dev = ensemble.std(axis=1, keepdims=True)
    jitter = np.random.normal(0, jitter_scale * std_dev, ensemble.shape)
    return ensemble + jitter


# Prior parameters
tot_assim_index = [[el] for el in range(len(TVD))]
v_corr = 50  # ft
grid_size = 128
Dh = 1.64042  # ft. (0.5 m)
log_rh_mean = 0.0
log_rh_std = 1.5

# Generate initial prior ensemble
pr = {
    'rh': (np.ones(grid_size) * log_rh_mean).reshape(-1, 1) + fast_gaussian(np.array([1, grid_size]),
                                                                              np.array([log_rh_std]),
                                                                              np.array([1, int(np.ceil(v_corr / Dh))]),
                                                                              num_samples=ne)
}

sample_cov = fast_gaussian(np.array([1, grid_size]), np.array([log_rh_std]), np.array([1, int(np.ceil(v_corr / Dh))]),
                           num_samples=50000)
Cm = np.cov(sample_cov, ddof=1)
prior_mean = sample_cov.mean(axis=1)

# Initialize with prior ensemble in log-space
current_log_rh_ensemble = pr['rh'].copy()
weights = np.ones(ne) / ne  # Initial uniform weights

# Main particle filter loop
for el in range(0, len(tot_assim_index), 1):  # Process every 10th step: 0, 10, 20, ...
    assim_index = tot_assim_index[el]
    # well position index is the closest cell to the current tvd
    well_pos_index = np.argmin(np.abs(cell_center_tvd - TVD[el]))
    
    Cd_row = Cd.iloc[el]
    Cd_vec = np.concatenate([np.array(cell[1])[[val[1] for val in mapping.values()]] for cell in Cd_row])
    data_vec = np.concatenate([data.iloc[el][dat][[val[1] for val in mapping.values()]] for dat in data_keys])
    
    data_real = data_vec #np.random.normal(loc=data_vec, scale=np.sqrt(Cd_vec))
    
    print(f"Assimilation step {el}")
    
    # Prediction step: simulate ensemble
    pred_ensemble, pred_by_tool = simulate_ensemble(np.exp(current_log_rh_ensemble).T, well_pos_index)  # shape (ne, n_data), (ne, n_tools, n_measurements)
    
    # Update step: calculate likelihood and update weights
    log_likelihood = calculate_log_likelihood(pred_ensemble, data_real, Cd_vec)
    
    # Update weights using log-sum-exp trick for numerical stability
    log_weights = np.log(weights) + log_likelihood
    max_log_weight = np.max(log_weights)
    weights = np.exp(log_weights - max_log_weight)
    weights /= np.sum(weights)  # Normalize
    
    plot = False  # Set to True to generate plots
    if plot:
        # ========== Data-Prediction Comparison Plots ==========
        n_data = len(data_real)
        data_labels = list(mapping.keys())
        
        # 1. Ensemble predictions with uncertainty bounds vs observations
        fig, ax = plt.subplots(figsize=(14, 6))
        pred_mean = np.mean(pred_ensemble, axis=0)
        pred_std = np.std(pred_ensemble, axis=0)
        pred_weighted_mean = np.average(pred_ensemble, axis=0, weights=weights)
        pred_weighted_std = np.sqrt(np.average((pred_ensemble - pred_weighted_mean)**2, axis=0, weights=weights))
        
        x = np.arange(n_data)
        ax.plot(x, data_real, 'ko-', label='Observed Data', linewidth=2, markersize=6)
        ax.plot(x, pred_mean, 'b-', label='Ensemble Mean (unweighted)', linewidth=1.5)
        ax.fill_between(x, pred_mean - 2*pred_std, pred_mean + 2*pred_std, alpha=0.2, color='blue', label='±2σ (unweighted)')
        ax.plot(x, pred_weighted_mean, 'r-', label='Ensemble Mean (weighted)', linewidth=1.5)
        ax.fill_between(x, pred_weighted_mean - 2*pred_weighted_std, pred_weighted_mean + 2*pred_weighted_std, 
                        alpha=0.2, color='red', label='±2σ (weighted)')
        ax.set_xlabel('Data Type')
        ax.set_ylabel('Value')
        ax.set_title(f'Predictions vs Observations (assim {el})')
        # ax.set_xticks(x)
        # ax.set_xticklabels(data_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'PF_data_pred_comparison_{data_type}_assim{el}.png', dpi=300)
        plt.close()
        
        # 2. Residuals plot with uncertainty
        fig, ax = plt.subplots(figsize=(14, 6))
        residuals_mean = data_real - pred_weighted_mean
        residuals_std = pred_weighted_std
        uncertainty_bound = 2 * np.sqrt(Cd_vec)  # 2σ observation uncertainty
        
        ax.errorbar(x, residuals_mean, yerr=residuals_std, fmt='o', label='Weighted Residuals ± 1σ', 
                    capsize=3, alpha=0.7)
        ax.fill_between(x, -uncertainty_bound, uncertainty_bound, alpha=0.2, color='green', 
                        label='±2σ Obs. Uncertainty')
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
        ax.set_xlabel('Data Type')
        ax.set_ylabel('Residual (Obs - Pred)')
        ax.set_title(f'Residuals (assim {el})')
        # ax.set_xticks(x)
        # ax.set_xticklabels(data_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'PF_residuals_{data_type}_assim{el}.png', dpi=300)
        plt.close()
        
        # 3. Scatter plot: observed vs predicted
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(data_real, pred_weighted_mean, c=weights.max() * np.ones(n_data), 
                s=50, alpha=0.6, cmap='viridis')
        
        # Add error bars for prediction uncertainty
        ax.errorbar(data_real, pred_weighted_mean, yerr=pred_weighted_std, fmt='none', 
                ecolor='gray', alpha=0.3, capsize=2)
        
        # 1:1 line
        lims = [min(data_real.min(), pred_weighted_mean.min()), 
                max(data_real.max(), pred_weighted_mean.max())]
        ax.plot(lims, lims, 'r--', alpha=0.5, linewidth=2, label='1:1 Line')
        
        ax.set_xlabel('Observed Data')
        ax.set_ylabel('Predicted Data (Weighted Mean)')
        ax.set_title(f'Observed vs Predicted (assim {el})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.savefig(f'PF_obs_vs_pred_{data_type}_assim{el}.png', dpi=300)
        plt.close()
        
        # 4. Normalized residuals (Z-scores)
        fig, ax = plt.subplots(figsize=(14, 6))
        z_scores = residuals_mean / np.sqrt(Cd_vec + pred_weighted_std**2)
        ax.bar(x, z_scores, color=['red' if abs(z) > 2 else 'blue' for z in z_scores], alpha=0.7)
        ax.axhline(y=2, color='r', linestyle='--', linewidth=1, label='±2σ threshold')
        ax.axhline(y=-2, color='r', linestyle='--', linewidth=1)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=1)
        ax.set_xlabel('Data Type')
        ax.set_ylabel('Normalized Residual (Z-score)')
        ax.set_title(f'Normalized Residuals (assim {el})')
        # ax.set_xticks(x)
        # ax.set_xticklabels(data_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'PF_normalized_residuals_{data_type}_assim{el}.png', dpi=300)
        plt.close()
        
        # 5. Ensemble spread visualization (spaghetti plot for subset)
        # fig, ax = plt.subplots(figsize=(14, 6))
        # # Plot a random subset of ensemble members
        # n_plot = min(10, ne)
        # indices_to_plot = np.random.choice(ne, n_plot, replace=False, p=weights)
        # for idx in indices_to_plot:
        #     ax.plot(x, pred_ensemble[idx], 'gray', alpha=0.1, linewidth=0.5)
        # ax.plot(x, pred_weighted_mean, 'b-', label='Weighted Mean', linewidth=2)
        # ax.plot(x, data_real, 'ro', label='Observed Data', markersize=6)
        # ax.set_xlabel('Data Index')
        # ax.set_ylabel('Value')
        # ax.set_title(f'Ensemble Spread (assim {el}, {n_plot} members)')
        # ax.legend()
        # ax.grid(True, alpha=0.3)
        # plt.tight_layout()
        # plt.savefig(f'PF_ensemble_spread_{data_type}_assim{el}.png', dpi=300)
        # plt.close()
        
        # 6. Histogram of residuals
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(residuals_mean / np.sqrt(Cd_vec), bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero residual')
        ax.set_xlabel('Normalized Residual')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of Normalized Residuals (assim {el})')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'PF_residual_histogram_{data_type}_assim{el}.png', dpi=300)
        plt.close()
        
        # ========== Tool-Specific Plots ==========
        # Reshape data_real and Cd_vec to (n_tools, n_measurements)
        n_tools = len(tool_info)
        n_measurements = len(data_labels)
        data_real_by_tool = data_real.reshape(n_tools, n_measurements)
        Cd_vec_by_tool = Cd_vec.reshape(n_tools, n_measurements)
        
        # Calculate unweighted statistics by tool
        pred_eqweight_mean_by_tool = np.mean(pred_by_tool, axis=0)  # shape (n_tools, n_measurements)
        pred_eqweight_std_by_tool = np.std(pred_by_tool, axis=0)
        
        # Calculate weighted statistics by tool
        pred_weighted_mean_by_tool = np.average(pred_by_tool, axis=0, weights=weights)  # shape (n_tools, n_measurements)
        pred_weighted_std_by_tool = np.sqrt(np.average((pred_by_tool - pred_weighted_mean_by_tool[np.newaxis, :, :])**2, 
                                                        axis=0, weights=weights))
        
        # 7. Tool-specific comparison plots (one subplot per tool)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for tool_idx, (freq, spacing) in enumerate(tool_info):
            ax = axes[tool_idx]
            x_meas = np.arange(n_measurements)
            
            # Data for this tool
            obs_tool = data_real_by_tool[tool_idx]
            pred_mean_tool = pred_eqweight_mean_by_tool[tool_idx]
            pred_mean_weighted_tool = pred_weighted_mean_by_tool[tool_idx]
            pred_std_tool = pred_eqweight_std_by_tool[tool_idx]
            uncertainty_tool = np.sqrt(Cd_vec_by_tool[tool_idx])
            
            # Plot
            ax.errorbar(x_meas, obs_tool, yerr=2*uncertainty_tool, fmt='ko-', 
                       label='Observed ±2σ', capsize=3, linewidth=2, markersize=6)
            ax.plot(x_meas, pred_mean_tool, 'r-', label='Predicted (un-weighted)', linewidth=1.5)
            ax.plot(x_meas, pred_mean_weighted_tool, 'b-', label='Predicted (weighted)', linewidth=1.5)
            ax.fill_between(x_meas, pred_mean_tool - 2*pred_std_tool, pred_mean_tool + 2*pred_std_tool,
                           alpha=0.3, color='red', label='±2σ pred')
            
            ax.set_xlabel('Measurement Type')
            ax.set_ylabel('Value')
            ax.set_title(f'Tool: {freq} @ {spacing}')
            ax.set_xticks(x_meas)
            ax.set_xticklabels(data_labels, rotation=45, ha='right', fontsize=8)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'PF_tool_specific_comparison_{data_type}_assim{el}.png', dpi=300)
        plt.close()
        
        # 8. Tool-specific residuals
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for tool_idx, (freq, spacing) in enumerate(tool_info):
            ax = axes[tool_idx]
            x_meas = np.arange(n_measurements)
            
            # Residuals for this tool
            obs_tool = data_real_by_tool[tool_idx]
            pred_mean_tool = pred_weighted_mean_by_tool[tool_idx]
            pred_std_tool = pred_weighted_std_by_tool[tool_idx]
            uncertainty_tool = np.sqrt(Cd_vec_by_tool[tool_idx])
            
            residuals_tool = obs_tool - pred_mean_tool
            z_scores_tool = residuals_tool / np.sqrt(Cd_vec_by_tool[tool_idx] + pred_std_tool**2)
            
            # Plot normalized residuals
            colors = ['red' if abs(z) > 2 else 'blue' for z in z_scores_tool]
            ax.bar(x_meas, z_scores_tool, color=colors, alpha=0.7)
            ax.axhline(y=2, color='r', linestyle='--', linewidth=1, alpha=0.5)
            ax.axhline(y=-2, color='r', linestyle='--', linewidth=1, alpha=0.5)
            ax.axhline(y=0, color='k', linestyle='-', linewidth=1)
            
            ax.set_xlabel('Measurement Type')
            ax.set_ylabel('Normalized Residual (Z-score)')
            ax.set_title(f'Tool: {freq} @ {spacing}')
            ax.set_xticks(x_meas)
            ax.set_xticklabels(data_labels, rotation=45, ha='right', fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim([-5, 5])
        
        plt.tight_layout()
        plt.savefig(f'PF_tool_specific_residuals_{data_type}_assim{el}.png', dpi=300)
        plt.close()
        
        # 9. Heatmap of residuals by tool and measurement
        fig, ax = plt.subplots(figsize=(12, 8))
        residuals_by_tool = data_real_by_tool - pred_weighted_mean_by_tool
        normalized_residuals = residuals_by_tool / np.sqrt(Cd_vec_by_tool)
        
        im = ax.imshow(normalized_residuals, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3)
        ax.set_xlabel('Measurement Type')
        ax.set_ylabel('Tool')
        ax.set_title(f'Normalized Residuals Heatmap (assim {el})')
        ax.set_xticks(np.arange(n_measurements))
        ax.set_xticklabels(data_labels, rotation=45, ha='right')
        ax.set_yticks(np.arange(n_tools))
        ax.set_yticklabels([f'{freq}@{spacing}' for freq, spacing in tool_info])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized Residual (Z-score)')
        
        # Add text annotations
        for i in range(n_tools):
            for j in range(n_measurements):
                text = ax.text(j, i, f'{normalized_residuals[i, j]:.1f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'PF_residuals_heatmap_{data_type}_assim{el}.png', dpi=300)
        plt.close()
        # ========== End Tool-Specific Plots ==========
        # ========== End Data-Prediction Comparison Plots ==========
    
    print(f"Weight statistics - Mean: {weights.mean():.6f}, Max: {weights.max():.6f}, Min: {weights.min():.6f}")
    
    # Calculate effective sample size
    ess = calculate_effective_sample_size(weights)
    ess_ratio = ess / ne
    print(f"Effective sample size: {ess:.1f} ({ess_ratio:.2%} of {ne})")

    # Calculate weighted mean and variance for plotting
    weighted_mean = np.sum(current_log_rh_ensemble * weights, axis=1)
    weighted_var = np.sum((current_log_rh_ensemble - weighted_mean.reshape(-1, 1)) ** 2 * weights, axis=1)
    weighted_std = np.sqrt(weighted_var)
    
    if plot:
        # Plot weighted ensemble mean of posterior
        plt.figure()
        plt.imshow(weighted_mean.reshape(grid_size, 1), aspect='auto')
        plt.colorbar()
        plt.title(f'Posterior weighted mean assim {el} - rh')
        plt.savefig(f'PF_rh_assim{el}_weighted_mean_{data_type}.png')
        plt.close()
        
        # Plot weighted ensemble standard deviation
        plt.figure()
        plt.imshow(weighted_std.reshape(grid_size, 1), aspect='auto')
        plt.colorbar()
        plt.title(f'Posterior weighted std assim {el} - rh')
        plt.savefig(f'PF_rh_assim{el}_weighted_std_{data_type}.png')
        plt.close()
        
        # Plot weight distribution
        plt.figure()
        plt.hist(weights, bins=50)
        plt.xlabel('Weight')
        plt.ylabel('Frequency')
        plt.title(f'Weight distribution assim {el}')
        plt.savefig(f'PF_rh_assim{el}_weights_{data_type}.png')
        plt.close()
    
    # save posterior ensemble and weights to file
    np.savez_compressed(f'PF_rh_assim{el}_posterior_ensemble_{data_type}.npz',
                        posterior_ensemble=current_log_rh_ensemble,
                        weights=weights)

    
    # Resampling step (if needed)
    # resample only if ESS is low
    if ess_ratio < resample_threshold:
        indices = systematic_resampling(weights)
        current_log_rh_ensemble = current_log_rh_ensemble[:, indices]
        # Add jitter to maintain ensemble diversity
        current_log_rh_ensemble = add_jitter(current_log_rh_ensemble, jitter_scale=0.02)
        weights = np.ones(ne) / ne  # uniform after resampling

    # Add process noise (always, independent of resampling)
    process_noise_scale = 0.1
    process_noise = fast_gaussian(
        np.array([1, grid_size]),
        np.array([log_rh_std * process_noise_scale]),
        np.array([1, int(np.ceil(v_corr / Dh))]),
        num_samples=ne
    )
    current_log_rh_ensemble += process_noise
    
    # old code for resampling and propagation
    ################################################################################
    # Resampling step (if needed)
    ################################################################################
    # if ess_ratio < resample_threshold:
    #     print(f"Resampling (ESS ratio {ess_ratio:.2%} < threshold {resample_threshold:.2%})")
    #     indices = systematic_resampling(weights)
    #     current_log_rh_ensemble = current_log_rh_ensemble[:, indices]
        
    #     # Add jitter to maintain ensemble diversity
    #     current_log_rh_ensemble = add_jitter(current_log_rh_ensemble, jitter_scale=0.02)
        
    #     # Reset weights to uniform
    #     weights = np.ones(ne) / ne
    #     print("Weights reset to uniform after resampling")
    
    #     # Propagation step for next assimilation
    # elif el < len(tot_assim_index) - 1:  # Don't generate for last step
    #     # Generate new ensemble from current posterior (proposal distribution)
    #     # Option 1: Sample with replacement based on weights and add diffusion
    #     indices = systematic_resampling(weights)
    #     current_log_rh_ensemble = current_log_rh_ensemble[:, indices]
        
    #     # Add diffusion/process noise to maintain diversity
    #     # This can be tuned based on the expected model error
    #     process_noise_scale = 0.1  # Adjust as needed
    #     process_noise = fast_gaussian(np.array([1, grid_size]),
    #                                 np.array([log_rh_std * process_noise_scale]),
    #                                 np.array([1, int(np.ceil(v_corr / Dh))]),
    #                                 num_samples=ne)
    #     current_log_rh_ensemble += process_noise
        
    #     # Reset weights for next step
    #     weights = np.ones(ne) / ne
print("Particle filter completed!")
