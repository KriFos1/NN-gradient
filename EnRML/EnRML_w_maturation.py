import os
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import torch
from NeuralSim.image_to_log import EMProxy
from ThreeDGiGEarth.common import h5_to_dict
from matplotlib import pyplot as plt
from pipt.update_schemes.update_methods_ns.approx_update import approx_update
from udar_proxi.utils import convert_bfield_to_udar

matplotlib.use('Agg')

# Reproducibility
np.random.seed(10)
torch.manual_seed(10)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Remove stale ensemble folders
for folder in os.listdir('.'):
    if folder.startswith('En_') and os.path.isdir(folder):
        import shutil
        shutil.rmtree(folder)

# Set paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
REFERENCE_MODEL = PROJECT_ROOT / 'inversion' / 'data' / 'Benchmark-3' / 'globalmodel.h5'

# Load observations and variances
data = pd.read_pickle(SCRIPT_DIR / 'data.pkl')
Cd = pd.read_pickle(SCRIPT_DIR / 'var.pkl')
data_keys = list(Cd.columns)

# Configure EnRML updater
upd = approx_update()
upd.ne = 100
upd.trunc_energy = 0.99
upd.keys_da = {}
upd.list_states = ['rh', 'rv']
upd.cell_index = None
upd.proj = (np.eye(upd.ne) - np.ones((upd.ne, upd.ne)) / upd.ne) / np.sqrt(upd.ne - 1)

data_type = 'UDAR'  # 'UDAR' or 'Bfield'

proxi_scalers = "https://gitlab.norceresearch.no/saly/image_to_log_weights/-/raw/master/em/{}.pth?ref_type=heads"
proxi_save_file = "https://gitlab.norceresearch.no/saly/image_to_log_weights/-/raw/master/em/checkpoint_770.pth?ref_type=heads"
nn_input_dict = {
    'input_shape': (3, 128),
    'output_shape': (6, 18),
    'checkpoint_path': proxi_save_file,
    'scaler': proxi_scalers,
}
NN_sim = EMProxy(**nn_input_dict).to(device)

with h5py.File(REFERENCE_MODEL, 'r') as f:
    ref_model = h5_to_dict(f)

TVD = ref_model['wellpath']['Z'][:, 0] * 3.28084  # feet
grid_size = 128
Dh = 1.64042  # ft (0.5 m)
well_pos_index = 64

def simulate_ensemble(param_ensemble: np.ndarray, well_pos_index: int) -> np.ndarray:
    """
    Run the NN simulator for an ensemble of resistivity models.
    """
    ne = param_ensemble.shape[0]
    resistivity = np.zeros((ne, 3, grid_size), dtype=np.float32)
    resistivity[:, 0, :] = param_ensemble[:, :grid_size]
    resistivity[:, 1, :] = param_ensemble[:, grid_size:]
    resistivity[:, 2, well_pos_index] = 1.0

    resistivity_tensor = torch.tensor(resistivity, dtype=torch.float32, device=device)
    nn_pred_tensor = NN_sim.image_to_log(resistivity_tensor)

    if data_type == 'UDAR':
        nn_pred = convert_bfield_to_udar(nn_pred_tensor[:, :, :10].detach().cpu().numpy())
    else:
        nn_pred = nn_pred_tensor[:, :, :10].detach().cpu().numpy()

    return nn_pred.reshape(ne, -1)


def calculate_ensemble_loss(pred_ensemble: np.ndarray, data_real: np.ndarray, param_ensemble: np.ndarray, Cd_vec: np.ndarray) -> np.ndarray:
    """
    Compute data misfit + prior misfit for bookkeeping.
    """
    residuals_data = data_real - pred_ensemble
    data_loss = 0.5 * np.sum((residuals_data ** 2) / Cd_vec, axis=1)

    residuals_theta = param_ensemble - mean_res.flatten()
    temp = np.linalg.solve(cov_res, residuals_theta.T)
    theta_loss = 0.5 * np.sum(residuals_theta.T * temp, axis=0)

    return data_loss + theta_loss


def truncated_cauchy(
    loc: float,
    scale: float,
    size: int,
    lower: float,
    upper: float,
) -> np.ndarray:
    """
    Draw samples from a Cauchy distribution truncated to [lower, upper].
    """
    remaining = size
    samples = []
    while remaining > 0:
        draws = loc + scale * np.random.standard_cauchy(remaining * 2)
        accepted = draws[(draws >= lower) & (draws <= upper)]
        if accepted.size == 0:
            continue
        take = min(accepted.size, remaining)
        samples.append(accepted[:take])
        remaining -= take
    return np.concatenate(samples)


def sample_hyperparameters(
    rh_mean_center: float,
    rv_mean_center: float,
    rh_mean_std: float,
    rv_mean_std: float,
    n_samples: int,
) -> np.ndarray:
    rh_samples = truncated_cauchy(
        loc=rh_mean_center,
        scale=rh_mean_std,
        size=n_samples,
        lower=0.1,
        upper=100.0,
    )
    rv_samples = truncated_cauchy(
        loc=rv_mean_center,
        scale=rv_mean_std,
        size=n_samples,
        lower=0.1,
        upper=100.0,
    )
    samples = np.stack([rh_samples, rv_samples], axis=1)
    return samples


def mean_likelihood(pred_ensemble: np.ndarray, data_obs: np.ndarray, Cd_vec: np.ndarray) -> float:
    residuals = data_obs - pred_ensemble
    log_like = -0.5 * np.sum((residuals ** 2) / Cd_vec, axis=1)
    log_like_max = np.max(log_like)
    return log_like_max
    #return float(np.exp(log_like_max) * np.mean(np.exp(log_like - log_like_max)))


def evaluate_hyperparameter_realizations(
    hyper_samples: np.ndarray,
    geostat_obj,
    cov_res: np.ndarray,
    grid_n: int,
    ensemble_size: int,
    prediction_return: bool = False,
) -> pd.DataFrame:
    results = []
    for rh_mean_i, rv_mean_i in hyper_samples:
        mean_r = np.concatenate([np.ones(grid_n) * rh_mean_i, np.ones(grid_n) * rv_mean_i])
        pr_r = geostat_obj.gen_real(mean_r, cov_res, ensemble_size)
        np.clip(pr_r, a_min=0.1, a_max=150, out=pr_r)

        pred_ensemble = simulate_ensemble(pr_r.T, well_pos_index)
        mean_like = mean_likelihood(pred_ensemble, data_real, Cd_vec)
        results.append(
            {
                'rh_mean': float(rh_mean_i),
                'rv_mean': float(rv_mean_i),
                'mean_likelihood': mean_like,
            }
        )
    if prediction_return:
        return pred_ensemble, pd.DataFrame(results)
    else:
        return pd.DataFrame(results)


# Prior
v_corr = 60  #ft
r_std = 1.5

from geostat.decomp import Cholesky
geostat = Cholesky()
Cr = geostat.gen_cov2d(grid_size,1,r_std**2, v_corr/Dh, 1,1,'exp')
# Introduce weak correlation between rh and rv fields
correlation_coeff = 0.75  #  Correlation coefficient (adjust between 0-1)
cross_cov = correlation_coeff * Cr  # Cross-covariance matrix

Cr_t = np.block([
    [Cr, cross_cov],
    [cross_cov, Cr]
])

rh_mean = 5.0
rv_mean = 5.0
rh_mean_std = 5.0
rv_mean_std = 5.0

tot_assim_index = [[el] for el in range(len(TVD))]

max_its = 20
    
# Assimilation loop
for el in range(0, len(tot_assim_index), 10):
    # Hyperparameter sampling and likelihood evaluation
    Cd_row = Cd.iloc[el]
    Cd_vec = np.concatenate([np.array(cell[1]) for cell in Cd_row])
    data_vec = np.concatenate([data.iloc[el][dat] for dat in data_keys])

    data_real = np.random.normal(loc=data_vec, scale=np.sqrt(Cd_vec))
    
    if el == 0:
        run_hyperparameter_scan = True
    else:
        # only run hyperparamet scan if pred_ensemble from last assim step does not cover data
        lower_bound = np.percentile(pred_ensemble, 1, axis=0)
        upper_bound = np.percentile(pred_ensemble, 99, axis=0)
        if np.all((data_real >= lower_bound) & (data_real <= upper_bound)):
            run_hyperparameter_scan = False
        else:
            run_hyperparameter_scan = True
    
    if run_hyperparameter_scan:
        print(f"Assimilation step {el}: Running hyperparameter scan...")
        # prediction_coverage = False
        # while not prediction_coverage:
        #     hyper_sample = sample_hyperparameters(
        #         rh_mean, rv_mean, rh_mean_std, rv_mean_std, 1
        #     )
        #     pred_ensemble, hyper_results = evaluate_hyperparameter_realizations(
        #         hyper_samples=hyper_sample,
        #         geostat_obj=geostat,
        #         cov_res=Cr_t,
        #         grid_n=grid_size,
        #         ensemble_size=upd.ne,
        #         prediction_return=True,
        #     )

        #     # Check if fan of ensemble predictions covers observed data, within 1-99 percentile
        #     lower_bound = np.percentile(pred_ensemble, 1, axis=0)
        #     upper_bound = np.percentile(pred_ensemble, 99, axis=0)
        #     if np.all((data_real >= lower_bound) & (data_real <= upper_bound)):
        #         prediction_coverage = True
        #         # only one sample, so take first
        #         best_rh_mean = hyper_sample[0,0]
        #         best_rv_mean = hyper_sample[0,1]

        hyper_sample_count = 500

        hyper_samples = sample_hyperparameters(rh_mean, rv_mean, rh_mean_std, rv_mean_std, hyper_sample_count)
        hyper_results = evaluate_hyperparameter_realizations(
            hyper_samples=hyper_samples,
            geostat_obj=geostat,
            cov_res=Cr_t,
            grid_n=grid_size,
            ensemble_size=upd.ne,
        )
        #hyper_results.to_csv('hyperparameter_likelihoods.csv', index=False)
        
        # Get the maximum mean likelihood and corresponding hyperparameters
        max_likelihood_idx = hyper_results['mean_likelihood'].idxmax()
        max_likelihood_value = hyper_results.loc[max_likelihood_idx, 'mean_likelihood']
        best_rh_mean = hyper_results.loc[max_likelihood_idx, 'rh_mean']
        best_rv_mean = hyper_results.loc[max_likelihood_idx, 'rv_mean']
    else:
        best_rh_mean = rh_mean
        best_rv_mean = rv_mean

    print(f"Assimilation step {el}: Best rh_mean = {best_rh_mean}, Best rv_mean = {best_rv_mean}")

    mean_r = np.concatenate([np.ones(grid_size)*best_rh_mean, np.ones(grid_size)*best_rv_mean])
    pr_r = geostat.gen_real(mean_r, Cr_t, upd.ne)

    np.clip(pr_r, a_min=0.1, a_max=150, out=pr_r)  # enforce physical bounds

    current_r_ensemble = pr_r.copy()
    tmp_r_ensemble = current_r_ensemble.copy()

    mean_res, cov_res = mean_r, Cr_t

    # start assimilation iterations
    upd.lam = 5e6    

    upd.real_obs_data = data_real.reshape(-1, 1)
    upd.scale_data = np.sqrt(Cd_vec)

    tot_loss_mean = []
    tot_loss_std = []
    tot_nrms_mean = []
    tot_nrms_std = []

    pred_ensemble = None
    for iteration in range(max_its):
        pred_ensemble = simulate_ensemble(tmp_r_ensemble.T, well_pos_index)
        ensemble_losses = calculate_ensemble_loss(pred_ensemble, data_real, tmp_r_ensemble.T, Cd_vec)

        tot_loss_mean.append(ensemble_losses.mean())
        tot_loss_std.append(ensemble_losses.std())

        residuals_data = data_real - pred_ensemble
        ensemble_nrms = np.sqrt(np.mean((residuals_data ** 2) / Cd_vec, axis=1))
        tot_nrms_mean.append(ensemble_nrms.mean())
        tot_nrms_std.append(ensemble_nrms.std())

        if iteration % 10 == 0:
            print(f"Assimilation step {el}, iteration {iteration}")
            print(f"Mean loss: {ensemble_losses.mean():.4f}, Std loss: {ensemble_losses.std():.4f}")
            print(f"Mean nRMS: {ensemble_nrms.mean():.4f}, Std nRMS: {ensemble_nrms.std():.4f}")

        if iteration > 0:
            if np.abs(tot_loss_mean[-1] - tot_loss_mean[-2]) < 1e-3:
                print("Converged based on loss change. Stopping iterations.")
                break
            if tot_loss_mean[-1] < tot_loss_mean[-2]:
                upd.lam *= 0.1
                current_r_ensemble = tmp_r_ensemble
            else:
                upd.lam *= 10

        upd.current_state = {'rh': current_r_ensemble[:grid_size, :],
                             'rv': current_r_ensemble[grid_size:, :]}
        upd.state_scaling = np.ones(2*grid_size)

        upd.pert_preddata = pred_ensemble.T @ upd.proj
        upd.aug_pred_data = pred_ensemble.T

        upd.update()
        tmp_r_ensemble = current_r_ensemble + upd.step

        # Enforce physical bounds
        np.clip(tmp_r_ensemble, a_min=0.1, a_max=150, out=tmp_r_ensemble)

        plot_diagnostics = True
        # plot the data and predictions
        if plot_diagnostics and (iteration== 0 or iteration == max_its-1):
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            # The vector of 48 values consist of 6 tools each with 8 UDAR measurements
            n_tools = 6
            measurement_filter_per_tool = [True, True, True, True, True, True, True, True]  # all measurements for UDAR
            n_measurements_per_tool = int(sum(measurement_filter_per_tool))  # number of measurements per tool after filtering
            measurement_names = [meas_name for c_indx,meas_name in enumerate(['USDP', 'USDA', 'UADP', 'UADA', 'UHRP', 'UHRA', 'UHAP', 'UHAA']) if measurement_filter_per_tool[c_indx]]
            tool_labels = [
                '6kHz @ 83ft',
                '12kHz @ 83ft',
                '24kHz @ 83ft',
                '24kHz @ 43ft',
                '48kHz @ 43ft',
                '96kHz @ 43ft',
            ]

            ensemble_by_tool = pred_ensemble.reshape(pred_ensemble.shape[0], n_tools, n_measurements_per_tool)
            # transpose so measurement index is first for simpler slicing per subplot
            ensemble_by_measurement = np.transpose(ensemble_by_tool, (0, 2, 1))
            data_by_measurement = np.asarray(data_real).reshape(n_tools, n_measurements_per_tool).T
            std_by_measurement = np.sqrt(Cd_vec).reshape(n_tools, n_measurements_per_tool).T
            x_pos = np.arange(n_tools)
            plotted_legend = False

            fig, axes_grid = plt.subplots(4, 2, figsize=(14, 16), sharex=True)
            fig.suptitle(f'NN Proxy Predictions, iteration {iteration}', fontsize=16, fontweight='bold')
            axes = axes_grid.flatten()
            for i in range(n_measurements_per_tool):
                ax = axes[i]
                lower = np.percentile(ensemble_by_measurement[:, i, :], 10, axis=0)
                upper = np.percentile(ensemble_by_measurement[:, i, :], 90, axis=0)
                median = np.median(ensemble_by_measurement[:, i, :], axis=0)
                ax.fill_between(
                    x_pos,
                    lower,
                    upper,
                    color='tab:blue',
                    alpha=0.2,
                    label='10-90 percentile' if not plotted_legend else "",
                )
                ax.plot(
                    x_pos,
                    median,
                    color='tab:blue',
                    marker='.',
                    linestyle='-',
                    linewidth=1,
                    markersize=6,
                    label='Median prediction' if not plotted_legend else "",
                )
                # Plot observed data with error bars (1 and 2 standard deviations)
                ax.errorbar(
                    x_pos,
                    data_by_measurement[i],
                    yerr=std_by_measurement[i],
                    color='tab:red',
                    marker='o',
                    linestyle='',
                    capsize=3,
                    capthick=1,
                    elinewidth=1.5,
                    label='Observed data (±1σ)' if not plotted_legend else "",
                )
                ax.errorbar(
                    x_pos,
                    data_by_measurement[i],
                    yerr=2*std_by_measurement[i],
                    color='tab:red',
                    marker='',
                    linestyle='',
                    capsize=0,
                    elinewidth=0.8,
                    alpha=0.4,
                    label='±2σ' if not plotted_legend else "",
                )
                ax.set_ylabel(measurement_names[i])
                if not plotted_legend:
                    ax.legend(loc='upper right', fontsize='small')
                    plotted_legend = True
                ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
            for ax in axes[-2:]:
                ax.set_xticks(x_pos)
                ax.set_xticklabels(tool_labels, rotation=30, ha='right')
            for ax in axes[-2:]:
                ax.set_xlabel('Tool (frequency @ spacing)')
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            plt.savefig(f'udar_data_vs_pred_assim_step_iter_{iteration}.png', dpi=300)
            plt.close()
    
    # Plot ensemble mean of posterior
    # plt.figure(); plt.imshow(current_log_rh_ensemble.mean(axis=1).reshape(grid_size,1), aspect='auto')
    # plt.colorbar(); plt.title(f'Posterior ensemble mean assim {el} - rh')
    # plt.savefig(f'rh_assim{el}_ensemble_mean_{data_type}.png'); plt.close()
    
    # # Plot ensemble standard deviation
    # plt.figure(); plt.imshow(current_log_rh_ensemble.std(axis=1).reshape(grid_size,1), aspect='auto')
    # plt.colorbar(); plt.title(f'Posterior ensemble std assim {el} - rh')
    # plt.savefig(f'rh_assim{el}_ensemble_std_{data_type}.png'); plt.close()

    # save posterior ensemble to file
    np.savez_compressed(f'rh_assim{el}_posterior_ensemble_{data_type}.npz', posterior_ensemble=current_r_ensemble, pred_ensemble=pred_ensemble)
    
    # Condition the ensemble for next assimilation step
    # This maintains spatial correlation while adding stochasticity
    rh_mean = current_r_ensemble[:grid_size, :].mean()
    rv_mean = current_r_ensemble[grid_size:, :].mean()
