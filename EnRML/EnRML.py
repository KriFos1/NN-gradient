import torch
from NeuralSim.image_to_log import EMProxy
from udar_proxi.utils import convert_bfield_to_udar

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

from pipt.update_schemes.update_methods_ns.approx_update import approx_update # load the approximate update method from PET

upd = approx_update()
# Required variables (without localization)
upd.ne = 1000  # number of ensemble members
upd.lam = 5e6  # LM parameter (0 for standard ES)
upd.trunc_energy = 0.99
upd.keys_da = {}  # No localization
upd.list_states = ['rh']
upd.cell_index = None

# State variables
#upd.current_state = {'rh': state_ensemble}  # shape (n_state, ne)
#upd.state_scaling = np.ones(n_state)  # or proper scaling

# Data
#upd.real_obs_data = obs_data  # shape (n_data, ne)
#upd.aug_pred_data = pred_data  # shape (n_data, ne)
#upd.scale_data = np.sqrt(data_variance)  # shape (n_data,)

# Perturbations
upd.proj = (np.eye(upd.ne) - np.ones((upd.ne, upd.ne))/upd.ne) / np.sqrt(upd.ne-1)
#upd.pert_preddata = pred_data @ upd.proj  # or compute scaled version


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
NN_sim = NN_sim.to(device)  # Move model to GPU

# read trajectory of well to ensure that the grid is aligned
T=np.loadtxt('../data/Benchmark-3/ascii/trajectory.DAT',comments='%')
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

data_type = 'UDAR'  # 'UDAR' or 'Bfield'
#mapping = nn_to_obs_bfield_mapping
mapping = nn_to_obs_udar_mapping

def custom_loss(predictions, data_real, theta,Cd):

    # Example: weighted mean squared error
    residuals_data = data_real - predictions.flatten()
    data_loss = 0.5 * np.sum((residuals_data ** 2) / Cd)
    
    residuals_theta = theta - mean_res.flatten()
    theta_loss = 0.5 * residuals_theta @ np.linalg.solve(cov_res, residuals_theta)
    
    loss = data_loss + theta_loss
    
    return loss

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
        Predictions for each ensemble member
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
        nn_pred = convert_bfield_to_udar(nn_pred_tensor[:, :, :10].detach().numpy()) # only first 10 are bfield components
    else:
        nn_pred = nn_pred_tensor[:,:,:10].detach().numpy()
    
    # Reshape to (ne, n_data)
    predictions = nn_pred.reshape(ne, -1)
 
    return predictions

def calculate_ensemble_loss(pred_ensemble, data_real, param_ensemble, Cd):
    """
    Calculate loss for each ensemble member.
    
    Parameters:
    -----------
    pred_ensemble : np.ndarray, shape (ne, n_data)
        Predictions for each ensemble member
    data_real : np.ndarray, shape (n_data,)
        Observed data
    param_ensemble : np.ndarray, shape (ne, grid_size)
        Parameter values for each ensemble member
    Cd : np.ndarray, shape (n_data,)
        Data variance
    
    Returns:
    --------
    losses : np.ndarray, shape (ne,)
        Loss for each ensemble member
    """
    ne = pred_ensemble.shape[0]
    losses = np.zeros(ne)
    
    for i in range(ne):
        # Data misfit term
        residuals_data = data_real - pred_ensemble[i]
        data_loss = 0.5 * np.sum((residuals_data ** 2) / Cd)
        
        # Prior term (assuming log-normal prior)
        residuals_theta = param_ensemble[i] - prior_mean.flatten()
        theta_loss = 0.5 * residuals_theta @ np.linalg.solve(Cm, residuals_theta)
        
        losses[i] = data_loss + theta_loss
    
    return losses


tot_assim_index = [[el] for el in range(len(TVD))]
v_corr = 50  #ft
grid_size = 128
Dh = 1.64042 #ft. (0.5 m)
log_rh_mean = 0.0
log_rh_std = 0.25
ne = upd.ne  # number of ensemble members
pr = {
    'rh': (np.ones(grid_size)*log_rh_mean).reshape(-1, 1)+ fast_gaussian(np.array([1,grid_size]),
                                                                            np.array([log_rh_std]),
                    np.array([1, int(np.ceil(v_corr / Dh))]),num_samples=ne)
            }

sample_cov = fast_gaussian(np.array([1,grid_size]),np.array([log_rh_std]),np.array([1, int(np.ceil(v_corr / Dh))]),num_samples=50000)
Cm = np.cov(sample_cov,ddof=1)
prior_mean = sample_cov.mean(axis=1)

# Initialize with prior ensemble in log-space
current_log_rh_ensemble = pr['rh'].copy()

#for el,assim_index in enumerate(tot_assim_index[:15]):
for el in range(0,100,1): #range(0, len(tot_assim_index), 10):  # Process every 10th step: 0, 10, 20, ...
    assim_index = tot_assim_index[el]
    # well position index is the closest cell to the current tvd
    well_pos_index = np.argmin(np.abs(cell_center_tvd - TVD[el]))

    Cd_row = Cd.iloc[el]
    Cd_vec = np.concatenate([np.array(cell[1])[[val[1] for val in mapping.values()]] for cell in Cd_row])
    data_vec = np.concatenate([data.iloc[el][dat][[val[1] for val in mapping.values()]] for dat in data_keys])

    data_real = np.random.normal(loc=data_vec, scale=np.sqrt(Cd_vec))

    upd.real_obs_data = data_real.reshape(-1,1)  # shape (n_data, ne)
    upd.scale_data = np.sqrt(Cd_vec)  # shape (n_data,)

    tot_loss = []
    
    for iteration in range(200):  # number of EnRML iterations per assimilation step
        print(f"Assimilation step {el}, iteration {iteration}")
        pred_ensemble = simulate_ensemble(np.exp(current_log_rh_ensemble).T, well_pos_index)  # shape (ne, n_data)
        
        # Calculate loss for each ensemble member
        ensemble_losses = calculate_ensemble_loss(pred_ensemble, data_real, np.exp(current_log_rh_ensemble).T, Cd_vec)
        print(f"Mean loss: {ensemble_losses.mean():.4f}, Std loss: {ensemble_losses.std():.4f}")

        upd.current_state = {'rh': current_log_rh_ensemble}  # shape (n_state, ne)
        upd.state_scaling = np.ones(grid_size)  # or proper scaling

        upd.pert_preddata = pred_ensemble.T @ upd.proj  # shape (n_data, ne)
        upd.aug_pred_data = pred_ensemble.T  # shape (n_data, ne)
        
        # Perform EnRML update
        upd.update()

        # deploy step to ensemble
        current_log_rh_ensemble += upd.step
    
    # Plot ensemble mean of posterior
    plt.figure(); plt.imshow(current_log_rh_ensemble.mean(axis=1).reshape(grid_size,1), aspect='auto')
    plt.colorbar(); plt.title(f'Posterior ensemble mean assim {el} - rh')
    plt.savefig(f'rh_assim{el}_ensemble_mean_{data_type}.png'); plt.close()
    
    # Plot ensemble standard deviation
    plt.figure(); plt.imshow(current_log_rh_ensemble.std(axis=1).reshape(grid_size,1), aspect='auto')
    plt.colorbar(); plt.title(f'Posterior ensemble std assim {el} - rh')
    plt.savefig(f'rh_assim{el}_ensemble_std_{data_type}.png'); plt.close()

    # save posterior ensemble to file
    np.savez_compressed(f'rh_assim{el}_posterior_ensemble_{data_type}.npz', posterior_ensemble=current_log_rh_ensemble)
    
    # Condition the ensemble for next assimilation step
    # This maintains spatial correlation while adding stochasticity
    if el < len(tot_assim_index) - 1:  # Don't generate for last step
 #       print(f"Generating conditioned ensemble for next assimilation step...")
        current_log_rh_ensemble = (np.ones(grid_size)*log_rh_mean).reshape(-1, 1)+ fast_gaussian(np.array([1,grid_size]),
                                                                            np.array([log_rh_std]),
                    np.array([1, int(np.ceil(v_corr / Dh))]),num_samples=ne)
 
