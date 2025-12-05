import torch
from NeuralSim.image_to_log import EMProxy

import os
# fix the seed for reproducability
import numpy as np
from geostat.gaussian_sim import fast_gaussian
import pandas as pd
from scipy.optimize import minimize, Bounds
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for parallel processing
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

np.random.seed(10)

# Source - https://stackoverflow.com/a
# Posted by joni
# Retrieved 2025-11-17, License - CC BY-SA 4.0

from scipy.optimize._optimize import MemoizeJac

class MemoizeJacHess(MemoizeJac):
    """ Decorator that caches the return vales of a function returning
        (fun, grad, hess) each time it is called. """

    def __init__(self, fun):
        super().__init__(fun)
        self.hess = None

    def _compute_if_needed(self, x, *args):
        if not np.all(x == self.x) or self._value is None or self.jac is None or self.hess is None:
            self.x = np.asarray(x).copy()
            self._value, self.jac, self.hess = self.fun(x, *args)

    def hessian(self, x, *args):
        self._compute_if_needed(x, *args)
        return self.hess


proxi_scalers = "https://gitlab.norceresearch.no/saly/image_to_log_weights/-/raw/master/em/{}.pth?ref_type=heads"
proxi_save_file = "https://gitlab.norceresearch.no/saly/image_to_log_weights/-/raw/master/em/checkpoint_770.pth?ref_type=heads"

np.random.seed(10)


# remove folders in current directory starting with 'En_'
for folder in os.listdir('.'):
    if folder.startswith('En_') and os.path.isdir(folder):
        import shutil
        shutil.rmtree(folder)


data = pd.read_pickle('data.pkl')
Cd = pd.read_pickle('var.pkl')
data_keys = list(Cd.columns)  # Extract column headers from Cd
datatype = [str(dat) for dat in data_keys] # make into strings
param_keys = ['rh']

nn_input_dict = {'input_shape': (3,128), # These are fixed
                      'output_shape':  (6,18),
                      'checkpoint_path': proxi_save_file,
                      'scaler': proxi_scalers
    }
NN_sim = EMProxy(**nn_input_dict)

# read trajectory of well to ensure that the grid is aligned
T=np.loadtxt('data/Benchmark-3/ascii/trajectory.DAT',comments='%')
TVD =T[:,1]

# The grid does not have any depth. The well position is controlled by the one-hot vector, and the assumption that each grid cell is 0.5 m (1.64042 ft) thick.
# Given the well trajectory, the grid cell size, and the initial value of the one-hot vector. We can update the one-hot vector to match the well position at each measurement point.
# One strict assumption is that tvd[-1] - tvd[0] < 128 * 1.64042 ft
# Assert thision
Dh = 1.64042 #ft. (0.5 m)
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

observed_data_order_bfield = ['real(xx)', 'real(xy)', 'real(xz)', 'real(yx)', 'real(yy)', 'real(yz)', 'real(zx)', 'real(zy)', 'real(zz)',
                             'img(xx)', 'img(xy)', 'img(xz)', 'img(yx)', 'img(yy)', 'img(yz)', 'img(zx)', 'img(zy)', 'img(zz)']

nn_to_obs_udar_mapping = {data_name: (10+nn_index, observed_data_order_udar.index(data_name)) for nn_index, data_name in enumerate(NN_data_names[10:])}
nn_to_obs_bfield_mapping = {data_name: (nn_index, observed_data_order_bfield.index(data_name)) for nn_index, data_name in enumerate(NN_data_names[:10])}

data_type = 'Bfield'  # 'UDAR' or 'Bfield'
mapping = nn_to_obs_bfield_mapping
#mapping = nn_to_obs_udar_mapping

def custom_loss(predictions, data_real, theta,Cd):

    # Example: weighted mean squared error
    residuals_data = data_real - predictions.flatten()
    data_loss = 0.5 * np.sum((residuals_data ** 2) / Cd)
    
    residuals_theta = theta - mean_res.flatten()
    theta_loss = 0.5 * residuals_theta @ np.linalg.solve(cov_res, residuals_theta)
    
    loss = data_loss + theta_loss
    
    return loss

def simulate_and_grad(param, data_real, Cd):

    resistivity = np.zeros((3, grid_size))
    resistivity[0,:] = param
    resistivity[1,:] = param
    resistivity[2,well_pos_index] = 1  # well position
    resistivity_tensor = torch.tensor(resistivity, dtype=torch.float32)

    nn_pred_tensor = NN_sim.image_to_log(resistivity_tensor)
    nn_jacobian = torch.autograd.functional.jacobian(NN_sim.image_to_log, resistivity_tensor)
    jacobian = nn_jacobian.detach().numpy()[0,:,[val[0] for val in mapping.values()],0,:].reshape(-1, grid_size)
    nn_pred = nn_pred_tensor.detach().numpy()[:,:,[val[0] for val in mapping.values()]].flatten()

    loss = custom_loss(nn_pred, data_real, param,Cd)
    #print(loss)

    grad = jacobian.T @ ((nn_pred - data_real) / Cd) + np.linalg.solve(cov_res, param - mean_res.flatten())

    # hessian_approx = jacobian.T @ (jacobian / Cd[:, np.newaxis]) + inv_cm

    return loss, grad #, hessian_approx

def lognormal_from_log_gaussian(mu, Sigma):
    """
    Compute mean and covariance of a log-normal vector X = exp(Y)
    where Y ~ N(mu, Sigma).

    Parameters
    ----------
    mu : (n,) array_like
        Mean vector of the Gaussian in log-space.
    Sigma : (n, n) array_like
        Covariance matrix of the Gaussian in log-space.

    Returns
    -------
    mean_X : (n,) ndarray
        Mean of the log-normal vector X.
    cov_X : (n, n) ndarray
        Covariance matrix of X.
    """
    mu = np.atleast_1d(mu)
    Sigma = np.atleast_2d(Sigma)

    if Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("Sigma must be a square matrix.")
    if Sigma.shape[0] != mu.shape[0]:
        raise ValueError("mu and Sigma dimensions are inconsistent.")

    # Mean of X_i = E[exp(Y_i)] = exp(mu_i + 0.5 * Sigma_ii)
    var_Y = np.diag(Sigma)
    mean_X = np.exp(mu + 0.5 * var_Y)

    # Cov(X_i, X_j) = E[X_i] E[X_j] (exp(Sigma_ij) - 1)
    # Use broadcasting with outer product and elementwise operations
    outer_mean = np.outer(mean_X, mean_X)
    exp_Sigma = np.exp(Sigma)
    cov_X = outer_mean * (exp_Sigma - 1.0)

    return mean_X, cov_X


tot_assim_index = [[el] for el in range(50)]

v_corr = 50  #ft
grid_size = 128
Dh = 1.64042 #ft. (0.5 m)
log_rh_mean = 0.0
log_rh_std = 0.25
ne = 200
pr = {
    'rh': (np.ones(grid_size)*log_rh_mean).reshape(-1, 1)+ fast_gaussian(np.array([1,grid_size]),
                                                                            np.array([log_rh_std]),
                    np.array([1, int(np.ceil(v_corr / Dh))]),num_samples=ne)
            }

sample_cov = fast_gaussian(np.array([1,grid_size]),np.array([log_rh_std]),np.array([1, int(np.ceil(v_corr / Dh))]),num_samples=50000)
Cm = np.cov(sample_cov,ddof=1)
#Cm = np.ones(grid_size)*(log_rh_std**2)  # prior covariance matrix

mean_res, cov_res = lognormal_from_log_gaussian(np.ones(grid_size)*log_rh_mean, Cm)

def generate_conditioned_ensemble(posterior_log_rh, Cm, ne, seed=42):
    """
    Generate new ensemble conditioned on posterior using Sequential Gaussian Simulation (Kriging-based).
    
    This uses kriging to condition unconditional realizations on the posterior ensemble statistics,
    preserving spatial correlation while introducing stochasticity for the next assimilation step.
    
    Parameters:
    -----------
    posterior_log_rh : (grid_size, ne) array
        Log-resistivity ensemble from previous assimilation step
    Cm : (grid_size, grid_size) array
        Prior covariance matrix
    ne : int
        Number of ensemble members
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    new_log_rh : (grid_size, ne) array
        New conditioned ensemble in log-space
    """
    grid_size = posterior_log_rh.shape[0]
    
    # Compute posterior statistics
    posterior_mean = posterior_log_rh.mean(axis=1)
    
    # Generate unconditional realizations from prior
    new_log_rh = np.zeros((grid_size, ne))
    rng = np.random.default_rng(seed)
    
    # Cholesky decomposition of prior covariance for unconditional simulation
    try:
        L = np.linalg.cholesky(Cm + 1e-8 * np.eye(grid_size))
    except np.linalg.LinAlgError:
        # Fallback to eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(Cm)
        eigvals = np.maximum(eigvals, 1e-8)
        L = eigvecs @ np.diag(np.sqrt(eigvals))
    
    # Generate unconditional realizations
    for i in range(ne):
        z = rng.standard_normal(grid_size)
        new_log_rh[:, i] = L @ z
    
    # Kriging: condition unconditional realizations on posterior mean
    # Using simple kriging: x_conditioned = x_uncond + C_cross @ C_data^-1 @ (data - mean)
    
    # For computational efficiency, use subset of grid points as conditioning data
    # Sample every n-th point or use reduced set
    conditioning_stride = max(1, grid_size // 64)  # Use ~64 conditioning points
    conditioning_indices = np.arange(0, grid_size, conditioning_stride)
    n_cond = len(conditioning_indices)
    
    # Extract covariance blocks
    C_data = Cm[np.ix_(conditioning_indices, conditioning_indices)]  # Cov between conditioning points
    C_cross = Cm[:, conditioning_indices]  # Cross-covariance
    
    # Kriging weights: K = C_cross @ C_data^-1
    try:
        K = C_cross @ np.linalg.inv(C_data + 1e-6 * np.eye(n_cond))
    except np.linalg.LinAlgError:
        # Use pseudo-inverse if singular
        K = C_cross @ np.linalg.pinv(C_data + 1e-6 * np.eye(n_cond))
    
    # Condition each realization
    for i in range(ne):
        # Kriging update: add correction based on difference at conditioning points
        uncond_at_cond = new_log_rh[conditioning_indices, i]
        target_at_cond = posterior_mean[conditioning_indices]
        residual = target_at_cond - uncond_at_cond
        
        # Apply kriging correction
        new_log_rh[:, i] = new_log_rh[:, i] + K @ residual
    
    return new_log_rh

def process_ensemble_member(ens_idx, el, assim_index, current_log_rh_ensemble):
    """Process a single ensemble member optimization"""
    # Use ensemble-specific resistivity from current ensemble
    resistivity = np.exp(current_log_rh_ensemble[:, ens_idx])
    
    # Extract the el-th row of Cd and concatenate variance values
    # Each element in Cd is ['ABS', [...]], we want only the [...] part
    Cd_row = Cd.iloc[assim_index]
    Cd_vec = np.concatenate([np.array(cell[1])[[val[1] for val in mapping.values()]] for cell in Cd_row])
    data_vec = np.concatenate([data.iloc[assim_index][dat][[val[1] for val in mapping.values()]] for dat in data_keys])

    # Use ensemble-specific random seed for reproducibility
    rng = np.random.default_rng(seed=10 + ens_idx)
    data_real = rng.normal(loc=data_vec, scale=np.sqrt(Cd_vec))

    res = minimize(simulate_and_grad,
                   x0=resistivity.flatten(),
                   args=(data_real, Cd_vec),
                   method='trust-constr',
                   jac=True,
                   bounds=Bounds(lb=0.5, ub=np.inf),
                   options={'verbose': 0,
                            'maxiter': 100,
                            'initial_tr_radius': 50.0,  # Larger initial trust region
                            'gtol': 1e-1,
                            'xtol': 1e-1
                            })
    
    # Save plots for this ensemble member
    # plt.figure(); plt.imshow(res.x.reshape(grid_size,1), aspect='auto')
    # plt.colorbar(); plt.title(f'Posterior ens {ens_idx} assim {el} - rh')
    # plt.savefig(f'rh_assim{el}_ens{ens_idx}_{data_type}.png'); plt.close()

    # # plot the difference
    # plt.figure(); plt.imshow((res.x - resistivity).reshape(grid_size,1), aspect='auto', 
    #                           cmap='seismic', vmin=-np.max(np.abs(res.x - resistivity)), 
    #                           vmax=np.max(np.abs(res.x - resistivity)))
    # plt.colorbar(); plt.title(f'Difference ens {ens_idx} assim {el} - rh')
    # plt.savefig(f'rh_diff_assim{el}_ens{ens_idx}_{data_type}.png'); plt.close()
    
    return ens_idx, res.x, res.fun, res.success

# Plot the initial ensemble mean
resistivity_mean = np.exp(pr['rh']).mean(axis=1)
plt.figure(); plt.imshow(resistivity_mean.reshape(grid_size,1), aspect='auto')
plt.colorbar(); plt.title('Initial ensemble mean - rh'); plt.savefig(f'rh_initial_mean_{data_type}.png'); plt.close()

# Initialize with prior ensemble in log-space
current_log_rh_ensemble = pr['rh'].copy()

#for el,assim_index in enumerate(tot_assim_index[:15]):
for el, assim_index in enumerate(tot_assim_index[::10]):  # Process multiple assimilation steps
    # well position index is the closest cell to the current tvd
    well_pos_index = np.argmin(np.abs(cell_center_tvd - TVD[el]))

    print(f"\nProcessing assimilation step {el} with {ne} ensemble members in parallel...")
    
    # Parallel execution over ensemble members
    n_jobs = min(100, ne, os.cpu_count())  # Use available CPU cores
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_ensemble_member)(ens_idx, el, assim_index, current_log_rh_ensemble) 
        for ens_idx in range(ne)
    )
    # results = list(map(
    #     lambda ens_idx: process_ensemble_member(ens_idx, el, assim_index, current_log_rh_ensemble),
    #     range(ne)
    # ))
    
    # Process results
    posterior_ensemble = np.zeros((grid_size, ne))
    for ens_idx, posterior, loss, success in results:
        posterior_ensemble[:, ens_idx] = posterior
        if not success:
            print(f"Warning: Optimization failed for ensemble member {ens_idx}")
    
    # Convert posterior to log-space for next iteration
    posterior_log_ensemble = np.log(posterior_ensemble)
    
    # Plot ensemble mean of posterior
    plt.figure(); plt.imshow(posterior_ensemble.mean(axis=1).reshape(grid_size,1), aspect='auto')
    plt.colorbar(); plt.title(f'Posterior ensemble mean assim {el} - rh')
    plt.savefig(f'rh_assim{el}_ensemble_mean_{data_type}.png'); plt.close()
    
    # Plot ensemble standard deviation
    plt.figure(); plt.imshow(posterior_ensemble.std(axis=1).reshape(grid_size,1), aspect='auto')
    plt.colorbar(); plt.title(f'Posterior ensemble std assim {el} - rh')
    plt.savefig(f'rh_assim{el}_ensemble_std_{data_type}.png'); plt.close()

    # save posterior ensemble to file
    np.savez_compressed(f'rh_assim{el}_posterior_ensemble_{data_type}.npz', posterior_ensemble=posterior_ensemble)
    
    # Condition the ensemble for next assimilation step
    # This maintains spatial correlation while adding stochasticity
    if el < len(tot_assim_index) - 1:  # Don't generate for last step
 #       print(f"Generating conditioned ensemble for next assimilation step...")
        current_log_rh_ensemble = (np.ones(grid_size)*log_rh_mean).reshape(-1, 1)+ fast_gaussian(np.array([1,grid_size]),
                                                                            np.array([log_rh_std]),
                    np.array([1, int(np.ceil(v_corr / Dh))]),num_samples=ne)
 #generate_conditioned_ensemble(
  #          posterior_log_ensemble, Cm, ne, seed=100 + el
  #      )
        
        # Plot conditioned ensemble mean to verify
#        conditioned_mean = np.exp(current_log_rh_ensemble).mean(axis=1)
#        plt.figure(); plt.imshow(conditioned_mean.reshape(grid_size,1), aspect='auto')
#        plt.colorbar(); plt.title(f'Conditioned ensemble mean for assim {el+1} - rh')
#        plt.savefig(f'rh_conditioned_{el+1}_ensemble_mean_{data_type}.png'); plt.close()
