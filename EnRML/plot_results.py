import matplotlib.pyplot as plt
import numpy as np
# convert from feet to meters
ft_to_m = 0.3048


# load all rh_assim{i}_posterior_ensemble_Bfield.npz files.
# They are stored using:
# np.savez_compressed(f'rh_assim{el}_posterior_ensemble_{data_type}.npz', posterior_ensemble=posterior_ensemble)
data_type = 'UDAR' 
T=np.loadtxt('../data/Benchmark-3/ascii/trajectory.DAT',comments='%')
ED =T[:,5] * ft_to_m # Convert to meters
tot_assim_index = [[el] for el in range(len(ED))]

tot_mean = []
tot_std = []
for el in range(0, len(tot_assim_index), 10):
    data = np.load(f'rh_assim{el}_posterior_ensemble_{data_type}.npz')
    posterior_ensemble = data['posterior_ensemble']
    # Calculate mean and std for this assimilation step
    tot_mean.append(np.mean(posterior_ensemble, axis=1))
    tot_std.append(np.std(posterior_ensemble, axis=1))

# Concatenate the mean and std for the posterior ensembles
mean_posterior = np.column_stack(tot_mean)
std_posterior = np.column_stack(tot_std)
fig, ax = plt.subplots(1, 2, figsize=(24, 6))
im1 = ax[0].imshow(mean_posterior, aspect='auto', origin='lower', interpolation='bilinear')
ax[0].set_title('Mean of Posterior Ensemble')
ax[0].set_xlabel('ED (m)')
selected_indices = list(range(0, len(tot_assim_index), 10))
selected_ed = [ED[i] for i in selected_indices]
ax[0].set_xticks(range(len(selected_indices)))
ax[0].set_xticklabels([f'{ed:.1f}' for ed in selected_ed], rotation=45)
ax[0].set_ylabel('State Variable Index')
ax[0].invert_yaxis()
fig.colorbar(im1, ax=ax[0])
im2 = ax[1].imshow(std_posterior, aspect='auto', origin='lower', interpolation='bilinear')
ax[1].set_title('Standard Deviation of Posterior Ensemble')
ax[1].set_xlabel('ED (m)')
ax[1].set_xticks(range(len(selected_indices)))
ax[1].set_xticklabels([f'{ed:.1f}' for ed in selected_ed], rotation=45)
ax[1].set_ylabel('State Variable Index')
ax[1].invert_yaxis()
fig.colorbar(im2, ax=ax[1])
plt.tight_layout()
plt.savefig(f'posterior_ensemble_stats_{data_type}.png', dpi=300)