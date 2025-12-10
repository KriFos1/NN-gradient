import torch
import numpy as np
import pandas as pd

from NeuralSim.image_to_log import EMProxy
from geostat.gaussian_sim import fast_gaussian

# Device and seeds
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(10)
print(f"Using device: {device}")

# Particle filter parameters
ne = 10000  # number of particles
resample_threshold = 0.5  # effective sample size threshold for triggering transport
use_tempering = True  # whether to use tempered optimal transport
n_tempering_steps = 5  # number of tempering iterations per assimilation step
target_ess_ratio = 0.7  # target ESS ratio for tempering

# Data and network setup (same mapping logic as particle.py)
data = pd.read_pickle('data.pkl')
Cd = pd.read_pickle('var.pkl')
data_keys = list(Cd.columns)
param_keys = ['rh']

proxi_scalers = "https://gitlab.norceresearch.no/saly/image_to_log_weights/-/raw/master/em/{}.pth?ref_type=heads"
proxi_save_file = "https://gitlab.norceresearch.no/saly/image_to_log_weights/-/raw/master/em/checkpoint_770.pth?ref_type=heads"

nn_input_dict = {
    'input_shape': (3, 128),
    'output_shape': (6, 18),
    'checkpoint_path': proxi_save_file,
    'scaler': proxi_scalers,
}
NN_sim = EMProxy(**nn_input_dict).to(device)

# Well path and grid setup
T = np.loadtxt('../data/Benchmark-3/ascii/trajectory.DAT', comments='%')
TVD = T[:, 1]
Dh = 1.64042  # ft (0.5 m)
assert TVD[-1] - TVD[0] < 128 * Dh, "The well trajectory exceeds the grid depth range."
grid_size = 128
well_start_index = 25
cell_center_tvd = TVD[0] + (np.arange(grid_size) - well_start_index) * Dh

# Observation mapping
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
observed_data_order_bfield = [
    'real(xx)', 'real(xy)', 'real(xz)', 'real(yx)', 'real(yy)', 'real(yz)', 'real(zx)', 'real(zy)', 'real(zz)',
    'img(xx)', 'img(xy)', 'img(xz)', 'img(yx)', 'img(yy)', 'img(yz)', 'img(zx)', 'img(zy)', 'img(zz)'
]
nn_to_obs_udar_mapping = {data_name: (10 + nn_index, observed_data_order_udar.index(data_name))
                          for nn_index, data_name in enumerate(NN_data_names[10:])}
nn_to_obs_bfield_mapping = {data_name: (nn_index, observed_data_order_bfield.index(data_name))
                            for nn_index, data_name in enumerate(NN_data_names[:10])}
data_type = 'UDAR'
mapping = nn_to_obs_udar_mapping
tool_info = [('6kHz', '83ft'), ('12kHz', '83ft'), ('24kHz', '83ft'),
             ('24kHz', '43ft'), ('48kHz', '43ft'), ('96kHz', '43ft')]


def simulate_ensemble(param_ensemble, well_pos_index):
    """Simulate an ensemble of resistivity models in parallel."""
    ne_local = param_ensemble.shape[0]
    resistivity = np.zeros((ne_local, 3, grid_size))
    resistivity[:, 0, :] = param_ensemble
    resistivity[:, 1, :] = param_ensemble
    resistivity[:, 2, well_pos_index] = 1.0

    resistivity_tensor = torch.tensor(resistivity, dtype=torch.float32, device=device)
    nn_pred_tensor = NN_sim.image_to_log(resistivity_tensor)
    nn_pred = nn_pred_tensor.detach().cpu().numpy()[:, :, [val[0] for val in mapping.values()]]
    predictions = nn_pred.reshape(ne_local, -1)
    return predictions, nn_pred


def calculate_log_likelihood(pred_ensemble, data_real, Cd_vec):
    """Gaussian log-likelihood for each particle."""
    residuals = data_real[None, :] - pred_ensemble
    return -0.5 * np.sum((residuals ** 2) / Cd_vec + np.log(2 * np.pi * Cd_vec), axis=1)


def calculate_effective_sample_size(weights):
    return 1.0 / np.sum(weights ** 2)


def add_jitter(ensemble, jitter_scale=0.01):
    std_dev = ensemble.std(axis=1, keepdims=True)
    jitter = np.random.normal(0, jitter_scale * std_dev, ensemble.shape)
    return ensemble + jitter


def compute_adaptive_temperature_increment(weights, target_ess_ratio, ne):
    """
    Compute temperature increment (0 to 1) that achieves target ESS.
    This represents how much to move from current weighted state toward uniform weights.
    
    Returns:
    --------
    alpha : float in [0, 1]
        Interpolation parameter: new_weights = (1-alpha)*weights + alpha*(1/ne)
    """
    temp_min = 0.0
    temp_max = 1.0
    
    for _ in range(20):  # Binary search iterations
        alpha = (temp_min + temp_max) / 2.0
        
        # Test interpolated weights
        test_weights = (1 - alpha) * weights + alpha * (np.ones_like(weights) / ne)
        test_weights /= np.sum(test_weights)
        
        ess = 1.0 / np.sum(test_weights ** 2)
        ess_ratio = ess / ne
        
        if abs(ess_ratio - target_ess_ratio) < 0.01:
            return alpha
        elif ess_ratio < target_ess_ratio:
            temp_max = alpha
        else:
            temp_min = alpha
    
    return temp_max


def compute_next_beta(log_likelihood, current_beta, target_ess_ratio, ne):
    """
    Compute the next tempering parameter beta for SMC.
    beta controls how much of the likelihood to apply: p(y|x)^beta
    
    Parameters:
    -----------
    log_likelihood : np.ndarray, shape (ne,)
        Log-likelihood for each particle
    current_beta : float
        Current tempering parameter (0 = prior, 1 = posterior)
    target_ess_ratio : float
        Target ESS ratio to maintain
    ne : int
        Number of particles
        
    Returns:
    --------
    beta_increment : float
        Amount to increase beta (result will be current_beta + beta_increment)
    """
    if current_beta >= 1.0:
        return 0.0
    
    beta_min = 0.0
    beta_max = 1.0 - current_beta
    
    for _ in range(20):  # Binary search
        beta_inc = (beta_min + beta_max) / 2.0
        
        # Compute what ESS would be with this increment
        # New weights ∝ old_weights * exp(beta_inc * log_likelihood)
        log_weights_inc = beta_inc * log_likelihood
        max_log = np.max(log_weights_inc)
        weights_inc = np.exp(log_weights_inc - max_log)
        weights_inc /= np.sum(weights_inc)
        
        ess = 1.0 / np.sum(weights_inc ** 2)
        ess_ratio = ess / ne
        
        if abs(ess_ratio - target_ess_ratio) < 0.02:
            return beta_inc
        elif ess_ratio < target_ess_ratio:
            beta_max = beta_inc
        else:
            beta_min = beta_inc
    
    return beta_max


def compute_forward_jacobian(mean_log_rh, well_pos_index):
    """Compute Jacobian of the forward map at the ensemble mean."""
    resistivity = np.zeros((1, 3, grid_size), dtype=np.float32)
    resistivity_values = np.exp(mean_log_rh)
    resistivity[0, 0, :] = resistivity_values
    resistivity[0, 1, :] = resistivity_values
    resistivity[0, 2, well_pos_index] = 1.0
    resistivity_tensor = torch.tensor(resistivity, device=device, requires_grad=True)

    nn_jacobian = torch.autograd.functional.jacobian(NN_sim.image_to_log, resistivity_tensor)
    jac = nn_jacobian.detach().cpu().numpy()

    # Use derivatives w.r.t. rh and rv channels (0 and 1), then apply chain rule d/d(log rh) = rh * d/d(rh)
    mapping_idx = [val[0] for val in mapping.values()]
    # Expected jac shape: (1, n_tools, 18, 1, 3, grid_size)
    jac_rh = jac[0, :, mapping_idx, 0, 0, :]  # take channels 0 and 1
    jac_rh *= resistivity_values[None, None, :]  # chain rule to log-space

    return jac_rh.reshape(-1, grid_size)


def compute_linear_transport_shift(log_ensemble, weights, pred_ensemble, data_real, Cd_vec, well_pos_index):
    """
    Linearized optimal-transport move based on the Jacobian of the forward model.
    """
    state_mean = np.average(log_ensemble, axis=1, weights=weights)
    centered = log_ensemble - state_mean[:, None]
    state_cov = (centered * weights) @ centered.T

    pred_mean = np.average(pred_ensemble, axis=0, weights=weights)
    innovation = data_real - pred_mean
    jacobian = compute_forward_jacobian(state_mean, well_pos_index)

    obs_cov = jacobian @ state_cov @ jacobian.T + np.diag(Cd_vec)
    obs_cov += 1e-6 * np.eye(obs_cov.shape[0])

    gain = state_cov @ jacobian.T
    correction = np.linalg.solve(obs_cov, innovation)
    shift = gain @ correction
    return shift, jacobian


def deterministic_transport(log_ensemble, weights, ordering_values, temperature=1.0):
    """
    1D monotone optimal transport with tempering.
    When temperature < 1.0, only partially moves towards equal weights.
    
    Parameters:
    -----------
    temperature : float
        Controls how much to move towards uniform weights (0=no move, 1=full transport)
    """
    m = weights.shape[0]
    target_mass = 1.0 / m

    sort_idx = np.argsort(ordering_values)
    inv_idx = np.argsort(sort_idx)
    sorted_weights = weights[sort_idx].copy()
    sorted_ensemble = log_ensemble[:, sort_idx]

    new_sorted = np.zeros_like(sorted_ensemble)
    j = 0
    remaining_mass = sorted_weights[j]
    for i in range(m):
        mass_needed = target_mass
        while mass_needed > 0 and j < m:
            mass = min(remaining_mass, mass_needed)
            if mass > 0:
                new_sorted[:, i] += (mass / target_mass) * sorted_ensemble[:, j]
                mass_needed -= mass
                remaining_mass -= mass
            if remaining_mass <= 1e-12:
                j += 1
                if j < m:
                    remaining_mass = sorted_weights[j]
    
    # Apply tempering: interpolate between original and transported ensemble
    new_ensemble = (1 - temperature) * log_ensemble + temperature * new_sorted[:, inv_idx]
    
    # Update weights: interpolate between original and uniform weights
    new_weights = (1 - temperature) * weights + temperature * (np.ones_like(weights) / m)
    new_weights /= np.sum(new_weights)  # renormalize
    
    return new_ensemble, new_weights


# Prior
tot_assim_index = [[el] for el in range(len(TVD))]
v_corr = 50  # ft
log_rh_mean = 0.0
log_rh_std = 1.5

pr = {
    'rh': (np.ones(grid_size) * log_rh_mean).reshape(-1, 1) + fast_gaussian(
        np.array([1, grid_size]),
        np.array([log_rh_std]),
        np.array([1, int(np.ceil(v_corr / Dh))]),
        num_samples=ne
    )
}

# Start the optimal transport particle filter
current_log_rh_ensemble = pr['rh'].copy()
weights = np.ones(ne) / ne

for el in range(0, len(tot_assim_index), 20):
    well_pos_index = np.argmin(np.abs(cell_center_tvd - TVD[el]))
    Cd_row = Cd.iloc[el]
    Cd_vec = np.concatenate([np.array(cell[1])[[val[1] for val in mapping.values()]] for cell in Cd_row])
    data_vec = np.concatenate([data.iloc[el][dat][[val[1] for val in mapping.values()]] for dat in data_keys])
    data_real = data_vec

    print(f"\nAssimilation step {el}")
    pred_ensemble, pred_by_tool = simulate_ensemble(np.exp(current_log_rh_ensemble).T, well_pos_index)

    log_likelihood = calculate_log_likelihood(pred_ensemble, data_real, Cd_vec)
    
    # Don't apply full likelihood yet - will be tempered
    ess_initial = calculate_effective_sample_size(weights)
    print(f"  Initial ESS: {ess_initial:.1f} ({ess_initial/ne:.2%})")

    # Iterative tempered transport with gradual likelihood incorporation
    if use_tempering:
        total_shift = np.zeros(current_log_rh_ensemble.shape[0])
        beta = 0.0  # Cumulative amount of likelihood applied (0=prior, 1=full posterior)
        
        for temp_iter in range(n_tempering_steps):
            # Find next beta that maintains target ESS
            beta_increment = compute_next_beta(log_likelihood, beta, target_ess_ratio, ne)
            
            if beta_increment < 1e-6:
                print(f"    Tempering step {temp_iter+1}: beta increment too small ({beta_increment:.6f})")
                break
            
            # Update weights by applying incremental likelihood
            log_weights = np.log(weights) + beta_increment * log_likelihood
            max_log_weight = np.max(log_weights)
            weights = np.exp(log_weights - max_log_weight)
            weights /= np.sum(weights)
            
            beta += beta_increment
            
            ess_after_reweight = calculate_effective_sample_size(weights)
            
            # Linear shift based on current weights
            shift, jacobian = compute_linear_transport_shift(
                current_log_rh_ensemble, weights, pred_ensemble, data_real, Cd_vec, well_pos_index
            )
            current_log_rh_ensemble = current_log_rh_ensemble + shift[:, None]
            total_shift += shift
            
            # Transport to equalize weights
            ordering_values = np.average(pred_ensemble, axis=1, weights=1.0 / (Cd_vec + 1e-12))
            current_log_rh_ensemble, weights = deterministic_transport(
                current_log_rh_ensemble, weights, ordering_values, temperature=1.0
            )
            
            ess_after_transport = calculate_effective_sample_size(weights)
            print(f"    Step {temp_iter+1}: beta={beta:.4f}, ESS: {ess_after_reweight:.1f} -> {ess_after_transport:.1f} ({ess_after_transport/ne:.2%})")
            
            # Re-evaluate predictions after transport
            if temp_iter < n_tempering_steps - 1 and beta < 0.999:
                pred_ensemble, _ = simulate_ensemble(np.exp(current_log_rh_ensemble).T, well_pos_index)
                log_likelihood = calculate_log_likelihood(pred_ensemble, data_real, Cd_vec)
            
            if beta >= 0.999:
                print(f"    Tempering complete: full likelihood applied (beta={beta:.4f})")
                break
        
        shift = total_shift  # Store cumulative shift for saving
    else:
        # Original non-tempered version
        shift, jacobian = compute_linear_transport_shift(
            current_log_rh_ensemble, weights, pred_ensemble, data_real, Cd_vec, well_pos_index
        )
        current_log_rh_ensemble = current_log_rh_ensemble + shift[:, None]
        
        ordering_values = np.average(pred_ensemble, axis=1, weights=1.0 / (Cd_vec + 1e-12))
        current_log_rh_ensemble, weights = deterministic_transport(
            current_log_rh_ensemble, weights, ordering_values, temperature=1.0
        )

    ess_final = calculate_effective_sample_size(weights)
    if ess_final / ne < resample_threshold:
        current_log_rh_ensemble = add_jitter(current_log_rh_ensemble, jitter_scale=0.01)
        print(f"  Applied jitter (ESS too low)")

    posterior_mean = np.mean(current_log_rh_ensemble, axis=1)
    posterior_std = np.std(current_log_rh_ensemble, axis=1)

    np.savez_compressed(
        f'OTPF_rh_assim{el}_posterior_ensemble_{data_type}.npz',
        posterior_ensemble=current_log_rh_ensemble,
        weights=weights,
        posterior_mean=posterior_mean,
        posterior_std=posterior_std,
        transport_shift=shift
        #jacobian=jacobian
    )

    process_noise_scale = 0.1
    process_noise = fast_gaussian(
        np.array([1, grid_size]),
        np.array([log_rh_std * process_noise_scale]),
        np.array([1, int(np.ceil(v_corr / Dh))]),
        num_samples=ne
    )
    current_log_rh_ensemble += process_noise

print("Optimal transport particle filter completed!")
