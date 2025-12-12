import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# convert from feet to meters
ft_to_m = 0.3048

# load all rh_assim{i}_posterior_ensemble_Bfield.npz files.
# They are stored using:
# np.savez_compressed(f'rh_assim{el}_posterior_ensemble_{data_type}.npz', posterior_ensemble=posterior_ensemble)
data_type = 'UDAR' 
T=np.loadtxt('../data/Benchmark-3/ascii/trajectory.DAT',comments='%')
TVD = T[:,1]
ED = T[:,5] * ft_to_m # Convert to meters
tot_assim_index = [[el] for el in range(len(ED))]

assim_points = range(0,len(tot_assim_index),1)

tot_mean = []
tot_std = []
tot_pred = []

print("Loading posterior ensembles and generating predictions...")
for el in assim_points:
    continue
    if el % 10 == 0:
        print(f"Processing assimilation point {el}...")
    
    data = np.load(f'rh_assim{el}_posterior_ensemble_{data_type}.npz')
    posterior_ensemble = data['posterior_ensemble']  # shape (grid_size, ne) in log space
    
    # Calculate mean and std for this assimilation step
    tot_mean.append(np.mean(posterior_ensemble, axis=1))
    tot_std.append(np.std(posterior_ensemble, axis=1))
    
    # load predictions
    tot_pred.append(data['pred_ensemble'])

# Concatenate the mean and std for the posterior ensembles
# mean_posterior = np.column_stack(tot_mean)
# std_posterior = np.column_stack(tot_std)
# fig, ax = plt.subplots(1, 2, figsize=(24, 6))
# im1 = ax[0].imshow(mean_posterior, aspect='auto', origin='lower', interpolation='bilinear')
# ax[0].set_title('Mean of Posterior Ensemble')
# ax[0].set_xlabel('ED (m)')
# selected_indices = list(assim_points)
# selected_ed = [ED[i] for i in selected_indices]
# ax[0].set_xticks(range(len(selected_indices)))
# ax[0].set_xticklabels([f'{ed:.1f}' for ed in selected_ed], rotation=45)
# ax[0].set_ylabel('State Variable Index')
# ax[0].invert_yaxis()
# fig.colorbar(im1, ax=ax[0])
# im2 = ax[1].imshow(std_posterior, aspect='auto', origin='lower', interpolation='bilinear')
# ax[1].set_title('Standard Deviation of Posterior Ensemble')
# ax[1].set_xlabel('ED (m)')
# ax[1].set_xticks(range(len(selected_indices)))
# ax[1].set_xticklabels([f'{ed:.1f}' for ed in selected_ed], rotation=45)
# ax[1].set_ylabel('State Variable Index')
# ax[1].invert_yaxis()
# fig.colorbar(im2, ax=ax[1])
# plt.tight_layout()
# plt.savefig(f'posterior_ensemble_stats_{data_type}.png', dpi=300)

# Plot predictions
import pandas as pd
import pickle

# Load observed data
data = pd.read_pickle('data.pkl')
Cd = pd.read_pickle('var.pkl')
data_keys = list(Cd.columns)  # Tools: [('6kHz','83ft'), ('12kHz','83ft'), etc.]

# UDAR data order for each tool (from EnRML.py)
observed_data_order_udar = [
    'USDP', 'USDA',
    'UADP', 'UADA',
    'UHRP', 'UHRA',
    'UHAP', 'UHAA'
]

# Data types to plot: Amplitude and Phase for USD, UAD, UHR, UHA
# Indices in observed_data_order_udar: USDP=0, USDA=1, UADP=2, UADA=3, UHRP=4, UHRA=5, UHAP=6, UHAA=7
plot_types = [
    ('USD', 1, 0),  # USDA (idx 1), USDP (idx 0)
    ('UAD', 3, 2),  # UADA (idx 3), UADP (idx 2)
    ('UHR', 5, 4),  # UHRA (idx 5), UHRP (idx 4)
    ('UHA', 7, 6)   # UHAA (idx 7), UHAP (idx 6)
]

n_tools = len(data_keys)
n_udar_types = 8

# Colors for each tool
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
tool_labels = [f'{freq} @ {spacing}' for freq, spacing in data_keys]

# Create figure with 4 panels
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
axes = axes.flatten()

for panel_idx, (title, amplitude_idx, phase_idx) in enumerate(plot_types):
    ax1 = axes[panel_idx]

    # Process each tool separately
    for tool_idx in range(n_tools):
        # Collect predictions for this tool across all assimilation points
        tool_pred_amplitude = []
        tool_pred_phase = []
        tool_obs_amplitude = []
        tool_obs_phase = []
        
        for el_idx, el in enumerate(assim_points):
            #pred = tot_pred[el_idx]  # shape: (n_data, ne) = (n_tools * 8, ne)
            
            # Reshape to (ne, n_tools, 8) to match original output format
            #pred_reshaped = pred.T.reshape(-1, n_tools, n_udar_types)  # (ne, n_tools, 8)
            
            # Extract predictions for this tool
            #amplitude_pred = pred_reshaped[:, tool_idx, amplitude_idx]  # shape (ne,)
            #phase_pred = pred_reshaped[:, tool_idx, phase_idx]  # shape (ne,)
            
            #tool_pred_amplitude.append(amplitude_pred)
            #tool_pred_phase.append(phase_pred)
            
            # Extract observed data for this tool
            obs_amplitude = data.iloc[el][data_keys[tool_idx]][amplitude_idx]
            obs_phase = data.iloc[el][data_keys[tool_idx]][phase_idx]
            tool_obs_amplitude.append(obs_amplitude)
            tool_obs_phase.append(obs_phase)
        
        # Convert to arrays
        #tool_pred_amplitude = np.array(tool_pred_amplitude)  # shape (n_assim_points, ne)
        #tool_pred_phase = np.array(tool_pred_phase)  # shape (n_assim_points, ne)
        
        # Calculate mean and std across ensemble members
        #mean_amplitude = np.mean(tool_pred_amplitude, axis=1)
        #std_amplitude = np.std(tool_pred_amplitude, axis=1)
        #mean_phase = np.mean(tool_pred_phase, axis=1)
        #std_phase = np.std(tool_pred_phase, axis=1)
        
        # Plot amplitude on left y-axis
        selected_ed = [ED[i] for i in assim_points]
        #line1 = ax1.plot(selected_ed, mean_amplitude, '-', color=colors[tool_idx], 
        #        linewidth=2, label=f'{tool_labels[tool_idx]} Amp')
        # ax1.fill_between(selected_ed, mean_amplitude - std_amplitude, mean_amplitude + std_amplitude, 
        #                  alpha=0.2, color=colors[tool_idx])
        ax1.plot(selected_ed, tool_obs_amplitude, 'o', color=colors[tool_idx], 
                markersize=4, alpha=0.7)
        
        # Plot phase on right y-axis with dashed line
        #line2 = ax1.plot(selected_ed, mean_phase, '--', color=colors[tool_idx], 
        #        linewidth=2, label=f'{tool_labels[tool_idx]} Phase')
        # ax2.fill_between(selected_ed, mean_phase - std_phase, mean_phase + std_phase, 
        #                  alpha=0.1, color=colors[tool_idx])
        ax1.plot(selected_ed, tool_obs_phase, 's', color=colors[tool_idx], 
                markersize=4, alpha=0.7)
    
    ax1.set_xlabel('ED (m)', fontsize=12)
    # ax1.set_ylabel(f'{title} Amplitude', fontsize=12, color='black')
    # ax1.set_ylabel(f'{title} Phase', fontsize=12, color='black')
    ax1.set_title(f'{title} Amplitude and Phase vs ED', fontsize=14)
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    #lines2, labels2 = ax2.get_legend_handles_labels()
    #ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='best')
    ax1.legend(lines1, labels1, fontsize=8, loc='best')
    
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='y', labelcolor='black')
    #ax2.tick_params(axis='y', labelcolor='black')

plt.tight_layout()
plt.savefig(f'predictions_comparison_{data_type}.png', dpi=300)
print(f"Saved predictions_comparison_{data_type}.png")
