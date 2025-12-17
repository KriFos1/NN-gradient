import torch
from NeuralSim.image_to_log import EMProxy
from udar_proxi.utils import convert_bfield_to_udar

import os
import logging
# fix the seed for reproducability
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('anneal_particle_filter.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set up device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")
from geostat.gaussian_sim import fast_gaussian
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for parallel processing
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(10)

# remove folders in current directory starting with 'APF_'
for folder in os.listdir('.'):
    if folder.startswith('APF_') and os.path.isdir(folder):
        import shutil
        shutil.rmtree(folder)

# Annealed Particle filter parameters
ne = 10000  # number of particles
resample_threshold = 0.1  # effective sample size threshold for resampling
n_tempering_steps = 1000  # number of annealing steps
tempering_schedule = 'geometric'  # 'geometric', 'linear', or 'adaptive'

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
mapping = nn_to_obs_udar_mapping
#mapping = nn_to_obs_bfield_mapping

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

    # convert NN bfield predictions to UDAR if needed
    if data_type == 'UDAR':
        nn_pred = convert_bfield_to_udar(nn_pred_tensor[:, :, :10].detach().cpu().numpy())[:,:,[val[1] for val in mapping.values()]] # only first 10 are bfield components
    else:
        nn_pred = nn_pred_tensor[:,:,:10].detach().numpy()
    
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


def compute_next_temperature(log_likelihood, current_temp, target_ess_ratio=0.5, ne=None):
    """
    Adaptively compute next temperature to maintain target ESS.
    
    Parameters:
    -----------
    log_likelihood : np.ndarray, shape (ne,)
        Log-likelihood for each particle
    current_temp : float
        Current temperature
    target_ess_ratio : float
        Target effective sample size ratio
    ne : int
        Number of particles
    
    Returns:
    --------
    next_temp : float
        Next temperature value
    """
    if ne is None:
        ne = len(log_likelihood)
    
    # Binary search for temperature that gives target ESS
    temp_min = current_temp
    temp_max = 1.0
    
    for _ in range(20):  # Max iterations for binary search
        temp_mid = (temp_min + temp_max) / 2.0
        delta_temp = temp_mid - current_temp
        
        # Compute incremental weights
        log_weights_inc = delta_temp * log_likelihood
        max_log_weight = np.max(log_weights_inc)
        weights_inc = np.exp(log_weights_inc - max_log_weight)
        weights_inc /= np.sum(weights_inc)
        
        # Calculate ESS ratio
        ess = 1.0 / np.sum(weights_inc ** 2)
        ess_ratio = ess / ne
        
        if abs(ess_ratio - target_ess_ratio) < 0.01:
            return temp_mid
        elif ess_ratio < target_ess_ratio:
            temp_max = temp_mid
        else:
            temp_min = temp_mid
    
    return temp_mid


def generate_tempering_schedule(n_steps, schedule_type='geometric'):
    """
    Generate tempering schedule.
    
    Parameters:
    -----------
    n_steps : int
        Number of tempering steps
    schedule_type : str
        Type of schedule: 'geometric', 'linear', or 'adaptive'
    
    Returns:
    --------
    temperatures : np.ndarray
        Temperature schedule from 0 to 1
    """
    if schedule_type == 'geometric':
        # Geometric progression
        temperatures = np.power(np.linspace(0, 1, n_steps + 1), 2)
    elif schedule_type == 'linear':
        # Linear progression
        temperatures = np.linspace(0, 1, n_steps + 1)
    else:  # adaptive
        # Will be computed adaptively
        temperatures = [0.0]  # Start with prior
    
    return temperatures


# Prior parameters
tot_assim_index = [[el] for el in range(len(TVD))]
v_corr = 5  # ft
grid_size = 128
Dh = 1.64042  # ft. (0.5 m)
log_rh_mean = 0.0
log_rh_std = 2.0

# Generate initial prior ensemble
pr = {
    'rh': (np.ones(grid_size) * log_rh_mean).reshape(-1, 1) + fast_gaussian(
        np.array([1, grid_size]),
        np.array([log_rh_std]),
        np.array([1, int(np.ceil(v_corr / Dh))]),
        num_samples=ne
    )
}
# log_rh_std = 1e-6
# pr = {
#     'rh': np.random.standard_cauchy(size=(grid_size, ne)) * log_rh_std + log_rh_mean
# }

# Initialize with prior ensemble in log-space
current_log_rh_ensemble = pr['rh'].copy()
weights = np.ones(ne) / ne  # Initial uniform weights

# Main annealed particle filter loop
for el in range(0,100,1): #range(0, len(tot_assim_index), 1):  # Process every 10th step: 0, 10, 20, ...
    assim_index = tot_assim_index[el]
    # well position index is the closest cell to the current tvd
    well_pos_index = np.argmin(np.abs(cell_center_tvd - TVD[el]))
    
    Cd_row = Cd.iloc[el]
    Cd_vec = np.concatenate([np.array(cell[1])[[val[1] for val in mapping.values()]] for cell in Cd_row])
    data_vec = np.concatenate([data.iloc[el][dat][[val[1] for val in mapping.values()]] for dat in data_keys])
    
    data_real = data_vec  # np.random.normal(loc=data_vec, scale=np.sqrt(Cd_vec))
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Assimilation step {el}")
    logger.info(f"{'='*60}")
    
    # Prediction step: simulate ensemble (only once, not in tempering loop)
    pred_ensemble, pred_by_tool = simulate_ensemble(np.exp(current_log_rh_ensemble).T, well_pos_index)
    
    # Calculate log-likelihood for all particles
    log_likelihood = calculate_log_likelihood(pred_ensemble, data_real, Cd_vec)
    
    # Generate or initialize tempering schedule
    if tempering_schedule == 'adaptive':
        temperatures = [0.0]
        current_temp = 0.0
        
        logger.info(f"\nAdaptive tempering schedule:")
        logger.info(f"Step 0: Temperature = {current_temp:.4f}")
        
        step = 0
        while current_temp < 1.0 and step < n_tempering_steps:
            next_temp = compute_next_temperature(log_likelihood, current_temp, 
                                                 target_ess_ratio=resample_threshold, ne=ne)
            if next_temp >= 1.0:
                next_temp = 1.0
            temperatures.append(next_temp)
            current_temp = next_temp
            step += 1
            logger.info(f"Step {step}: Temperature = {current_temp:.4f}")
        
        temperatures = np.array(temperatures)
    else:
        temperatures = generate_tempering_schedule(n_tempering_steps, tempering_schedule)
        logger.info(f"\n{tempering_schedule.capitalize()} tempering schedule:")
        for i, temp in enumerate(temperatures):
            logger.info(f"Step {i}: Temperature = {temp:.4f}")
    
    # Annealing loop
    logger.info(f"\nAnnealing iterations:")
    ess_history = []
    
    for step in range(1, len(temperatures)):
        temp_prev = temperatures[step - 1]
        temp_curr = temperatures[step]
        delta_temp = temp_curr - temp_prev
        
        # Update weights with incremental likelihood
        log_weights = np.log(weights) + delta_temp * log_likelihood
        max_log_weight = np.max(log_weights)
        weights = np.exp(log_weights - max_log_weight)
        weights /= np.sum(weights)  # Normalize
        
        # Calculate ESS
        ess = calculate_effective_sample_size(weights)
        ess_ratio = ess / ne
        ess_history.append(ess)
        
        logger.info(f"  Step {step}/{len(temperatures)-1}: temp={temp_curr:.4f}, "
              f"ESS={ess:.1f} ({ess_ratio:.2%}), "
              f"w_max={weights.max():.6f}, w_min={weights.min():.6f}")
        
        # Resample if ESS is too low
        if ess_ratio < resample_threshold:
            logger.info(f"    → Resampling (ESS ratio {ess_ratio:.2%} < threshold {resample_threshold:.2%})")
            indices = systematic_resampling(weights)
            current_log_rh_ensemble = current_log_rh_ensemble[:, indices]
            
            # Add jitter to maintain diversity
            current_log_rh_ensemble = add_jitter(current_log_rh_ensemble, jitter_scale=0.01)
            
            # Update predictions after resampling
            pred_ensemble, pred_by_tool = simulate_ensemble(np.exp(current_log_rh_ensemble).T, well_pos_index)
            log_likelihood = calculate_log_likelihood(pred_ensemble, data_real, Cd_vec)
            
            # Reset weights
            weights = np.ones(ne) / ne
            logger.info(f"    → Weights reset to uniform")
    
    logger.info(f"\nFinal weight statistics:")
    logger.info(f"  Mean: {weights.mean():.6f}, Max: {weights.max():.6f}, Min: {weights.min():.6f}")
    logger.info(f"  Final ESS: {calculate_effective_sample_size(weights):.1f} ({calculate_effective_sample_size(weights)/ne:.2%})")
    
    # Plotting section
    plot = False  # Set to True to enable plotting
    if plot:
        # ========== Data-Prediction Comparison Plots ==========
        n_data = len(data_real)
        data_labels = list(mapping.keys())
        
        # Calculate statistics
        pred_mean = np.mean(pred_ensemble, axis=0)
        pred_std = np.std(pred_ensemble, axis=0)
        pred_weighted_mean = np.average(pred_ensemble, axis=0, weights=weights)
        pred_weighted_std = np.sqrt(np.average((pred_ensemble - pred_weighted_mean)**2, axis=0, weights=weights))
        
        # 1. Predictions vs Observations
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(n_data)
        ax.plot(x, data_real, 'ko-', label='Observed Data', linewidth=2, markersize=6)
        ax.plot(x, pred_mean, 'b-', label='Ensemble Mean (unweighted)', linewidth=1.5)
        ax.fill_between(x, pred_mean - 2*pred_std, pred_mean + 2*pred_std, alpha=0.2, color='blue', label='±2σ (unweighted)')
        ax.plot(x, pred_weighted_mean, 'r-', label='Ensemble Mean (weighted)', linewidth=1.5)
        ax.fill_between(x, pred_weighted_mean - 2*pred_weighted_std, pred_weighted_mean + 2*pred_weighted_std, 
                        alpha=0.2, color='red', label='±2σ (weighted)')
        ax.set_xlabel('Data Type')
        ax.set_ylabel('Value')
        ax.set_title(f'Annealed PF: Predictions vs Observations (assim {el})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'APF_data_pred_comparison_{data_type}_assim{el}.png', dpi=300)
        plt.close()
        
        # 2. ESS evolution during annealing
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, len(ess_history)+1), ess_history, 'bo-', linewidth=2, markersize=8)
        ax.axhline(y=ne*resample_threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({resample_threshold:.1%})')
        ax.set_xlabel('Annealing Step')
        ax.set_ylabel('Effective Sample Size')
        ax.set_title(f'ESS Evolution During Annealing (assim {el})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'APF_ess_evolution_{data_type}_assim{el}.png', dpi=300)
        plt.close()
        
        # 3. Temperature schedule
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(len(temperatures)), temperatures, 'go-', linewidth=2, markersize=8)
        ax.set_xlabel('Annealing Step')
        ax.set_ylabel('Temperature')
        ax.set_title(f'Temperature Schedule (assim {el})')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.05, 1.05])
        plt.tight_layout()
        plt.savefig(f'APF_temperature_schedule_{data_type}_assim{el}.png', dpi=300)
        plt.close()
        
        # ========== Tool-Specific Plots ==========
        n_tools = len(tool_info)
        n_measurements = len(data_labels)
        data_real_by_tool = data_real.reshape(n_tools, n_measurements)
        Cd_vec_by_tool = Cd_vec.reshape(n_tools, n_measurements)
        
        # Calculate statistics by tool
        pred_mean_by_tool = np.mean(pred_by_tool, axis=0)
        pred_std_by_tool = np.std(pred_by_tool, axis=0)
        pred_weighted_mean_by_tool = np.average(pred_by_tool, axis=0, weights=weights)
        pred_weighted_std_by_tool = np.sqrt(np.average((pred_by_tool - pred_weighted_mean_by_tool[np.newaxis, :, :])**2, 
                                                        axis=0, weights=weights))
        
        # 4. Tool-specific comparison plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for tool_idx, (freq, spacing) in enumerate(tool_info):
            ax = axes[tool_idx]
            x_meas = np.arange(n_measurements)
            
            obs_tool = data_real_by_tool[tool_idx]
            pred_mean_tool = pred_weighted_mean_by_tool[tool_idx]
            pred_std_tool = pred_weighted_std_by_tool[tool_idx]
            uncertainty_tool = np.sqrt(Cd_vec_by_tool[tool_idx])
            
            ax.errorbar(x_meas, obs_tool, yerr=2*uncertainty_tool, fmt='ko-', 
                       label='Observed ±2σ', capsize=3, linewidth=2, markersize=6)
            ax.plot(x_meas, pred_mean_tool, 'r-', label='Predicted (weighted)', linewidth=1.5)
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
        plt.savefig(f'APF_tool_specific_comparison_{data_type}_assim{el}.png', dpi=300)
        plt.close()
        
        # 5. Residuals heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        residuals_by_tool = data_real_by_tool - pred_weighted_mean_by_tool
        normalized_residuals = residuals_by_tool / np.sqrt(Cd_vec_by_tool)
        
        im = ax.imshow(normalized_residuals, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3)
        ax.set_xlabel('Measurement Type')
        ax.set_ylabel('Tool')
        ax.set_title(f'Annealed PF: Normalized Residuals Heatmap (assim {el})')
        ax.set_xticks(np.arange(n_measurements))
        ax.set_xticklabels(data_labels, rotation=45, ha='right')
        ax.set_yticks(np.arange(n_tools))
        ax.set_yticklabels([f'{freq}@{spacing}' for freq, spacing in tool_info])
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized Residual (Z-score)')
        
        for i in range(n_tools):
            for j in range(n_measurements):
                text = ax.text(j, i, f'{normalized_residuals[i, j]:.1f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'APF_residuals_heatmap_{data_type}_assim{el}.png', dpi=300)
        plt.close()
    
    # Calculate weighted mean and variance for state
    weighted_mean = np.sum(current_log_rh_ensemble * weights, axis=1)
    weighted_var = np.sum((current_log_rh_ensemble - weighted_mean.reshape(-1, 1)) ** 2 * weights, axis=1)
    weighted_std = np.sqrt(weighted_var)
    
    if plot:
        # Plot posterior state
        plt.figure()
        plt.imshow(weighted_mean.reshape(grid_size, 1), aspect='auto')
        plt.colorbar()
        plt.title(f'Annealed PF: Posterior weighted mean assim {el} - rh')
        plt.savefig(f'APF_rh_assim{el}_weighted_mean_{data_type}.png')
        plt.close()
        
        plt.figure()
        plt.imshow(weighted_std.reshape(grid_size, 1), aspect='auto')
        plt.colorbar()
        plt.title(f'Annealed PF: Posterior weighted std assim {el} - rh')
        plt.savefig(f'APF_rh_assim{el}_weighted_std_{data_type}.png')
        plt.close()
        
        # Plot weight distribution
        plt.figure()
        plt.hist(weights, bins=50)
        plt.xlabel('Weight')
        plt.ylabel('Frequency')
        plt.title(f'Annealed PF: Weight distribution assim {el}')
        plt.savefig(f'APF_rh_assim{el}_weights_{data_type}.png')
        plt.close()
    
    # Save results
    np.savez_compressed(f'APF_rh_assim{el}_posterior_ensemble_{data_type}.npz',
                        posterior_ensemble=current_log_rh_ensemble,
                        weights=weights,
                        temperatures=temperatures,
                        ess_history=np.array(ess_history))
    
    # Propagation for next assimilation step
    if el < len(tot_assim_index) - 1:
        indices = systematic_resampling(weights)
        current_log_rh_ensemble = current_log_rh_ensemble[:, indices]
        
        # Add process noise
        process_noise_scale = 0.05
        process_noise = fast_gaussian(np.array([1, grid_size]),
                                      np.array([log_rh_std * process_noise_scale]),
                                      np.array([1, int(np.ceil(v_corr / Dh))]),
                                      num_samples=ne)
        current_log_rh_ensemble += process_noise
        
        # Reset weights
        weights = np.ones(ne) / ne

logger.info("\nAnnealed particle filter completed!")
