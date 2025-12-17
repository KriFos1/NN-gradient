import torch
import numpy as np
import copy
from scipy.interpolate import RegularGridInterpolator
from NeuralSim.image_to_log import EMProxy
from udar_proxi.utils import convert_bfield_to_udar
import h5py
from ThreeDGiGEarth.common import h5_to_dict
from EMsim.EM import UTA1D
from EMsim.EM import UTA2D
import pandas as pd
import os

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load NN model
proxi_scalers = "https://gitlab.norceresearch.no/saly/image_to_log_weights/-/raw/master/em/{}.pth?ref_type=heads"
proxi_save_file = "https://gitlab.norceresearch.no/saly/image_to_log_weights/-/raw/master/em/checkpoint_770.pth?ref_type=heads"

# Delete all En_* folders
for folder in os.listdir('.'):
    if folder.startswith('En_') and os.path.isdir(folder):
        import shutil
        shutil.rmtree(folder)

nn_input_dict = {
    'input_shape': (3, 128),
    'output_shape': (6, 18),
    'checkpoint_path': proxi_save_file,
    'scaler': proxi_scalers
}
NN_sim = EMProxy(**nn_input_dict)
NN_sim = NN_sim.to(device)

# Configuration
grid_size = 128
cell_thickness_m = 0.5  # meters
cell_thickness_ft = cell_thickness_m * 3.28084  # feet
well_cell_index = 64  # Place well at cell 64
out_of_region_value = 10.0  # Default resistivity for out-of-region values
data_type = 'UDAR'  # 'UDAR' or 'Bfield'

#
# Load trajectory

# Load 1D simulator
# initialize UTA simulator
UTA_input_dict = {'toolflag': 0,
            #'datasign': 0, # Bfield
            'datasign': 3, # UDAR
            'anisoflag': 1,
            'toolsetting': 'Benchmark-3/ascii/tool.inp',
            'trajectory': 'Benchmark-3/ascii/trajectory.DAT',
            'reference_model': '/home/AD.NORCERESEARCH.NO/krfo/CodeProjects/DISTINGUISH/Jacobian/inversion/data/Benchmark-3/globalmodel.h5',
            'parallel': 1,
            'map':{'ratio': [1,3]},
            #'surface_depth': [TVD[0]+(cell_thickness_ft/2) - cell_thickness_ft*np.ceil(grid_size/2) + cell_thickness_ft*i for i in range(grid_size+1)]
        }

with h5py.File(UTA_input_dict['reference_model'], "r") as f:
    ref_model = h5_to_dict(f)
TVD = ref_model['wellpath']['Z'][:,0] * 3.28084  # in feet
MD = ref_model['wellpath']['Distance'][:,0] * 3.28084  # in feet

UTA_input_dict['surface_depth'] = [TVD[0]+(cell_thickness_ft/2) - cell_thickness_ft*np.ceil(grid_size/2) + cell_thickness_ft*i for i in range(grid_size+1)]

UTA1D_input = {key: UTA_input_dict[key] for key in ['toolflag', 'datasign', 'anisoflag','reference_model','map',
                                                        'toolsetting', 'trajectory','surface_depth']}
sim_info = {
    'obsname': 'tvd',
    'assimindex': [[0]],
    'datatype': ["('6kHz','83ft')","('12kHz','83ft')","('24kHz','83ft')",
                "('24kHz','43ft')","('48kHz','43ft')","('96kHz','43ft')"]
}

UTA1D_sim = UTA1D({**UTA1D_input, **sim_info})
UTA1D_sim.setup_fwd_run(redund_sim=None)


UTA2D_input = {key: UTA_input_dict[key] for key in ['toolflag', 'datasign', 'anisoflag','reference_model','map',
                                                        'toolsetting', 'trajectory']}
UTA2D_input['dims'] = (31,1,grid_size)
UTA2D_input['dX'] = cell_thickness_ft*10
UTA2D_input['dY'] = cell_thickness_ft*10
UTA2D_input['dZ'] = cell_thickness_ft
UTA2D_input['shift'] = {'x':0,
                        'y':0,
                        'z':0}

UTA2D_sim = UTA2D({**UTA2D_input, **sim_info})
UTA2D_sim.setup_fwd_run(redund_sim=None)


def create_proxy_grid(well_z_m, grid_size, cell_thickness_m, well_cell_index):
    """
    Create proxy grid centered at the logging point.
    
    Parameters:
    -----------
    well_z_m : float
        Z-coordinate of the logging point in meters
    grid_size : int
        Number of grid cells (128)
    cell_thickness_m : float
        Thickness of each cell in meters
    well_cell_index : int
        Index where well is placed (64)
    
    Returns:
    --------
    proxy_z_centers : np.ndarray, shape (128,)
        Z-coordinates of proxy grid cell centers in meters
    """
    # Cell centers: well is at center of cell well_cell_index
    # Cell indices go from 0 to grid_size-1
    proxy_z_centers = well_z_m + (np.arange(grid_size) - well_cell_index) * cell_thickness_m
    
    return proxy_z_centers


def interpolate_formation_to_proxy(model, well_x_m, well_y_m, well_z_m, 
                                   proxy_z_centers, out_of_region_value=10.0):
    """
    Interpolate formation model (Rh, Rv) onto proxy grid.
    
    Parameters:
    -----------
    model : dict
        Dictionary with 'formation' key containing x, y, z, Rh, Rv arrays
    well_x_m, well_y_m, well_z_m : float
        Well position in meters
    proxy_z_centers : np.ndarray, shape (128,)
        Z-coordinates of proxy grid cell centers
    out_of_region_value : float
        Default value for out-of-region points
    
    Returns:
    --------
    proxy_rh : np.ndarray, shape (128,)
        Rh values on proxy grid
    proxy_rv : np.ndarray, shape (128,)
        Rv values on proxy grid
    """
    formation = model['formation']
    
    # Get unique coordinates and reshape Rh, Rv
    x_unique = np.unique(formation['x'])
    y_unique = np.unique(formation['y'])
    z_unique = np.unique(formation['z'])
    
    nx, ny, nz = len(x_unique), len(y_unique), len(z_unique)
    
    # Reshape Rh and Rv to 3D grid
    rh_3d = formation['Rh'].reshape((nz, ny, nx), order='C').transpose(2, 1, 0)
    rv_3d = formation['Rv'].reshape((nz, ny, nx), order='C').transpose(2, 1, 0)
    
    # Create interpolators
    rh_interpolator = RegularGridInterpolator(
        (x_unique, y_unique, z_unique), 
        rh_3d,
        method='linear',
        bounds_error=False,
        fill_value=out_of_region_value
    )
    
    rv_interpolator = RegularGridInterpolator(
        (x_unique, y_unique, z_unique),
        rv_3d,
        method='linear',
        bounds_error=False,
        fill_value=out_of_region_value
    )
    
    # Query points: same x, y for all proxy cells, varying z
    query_points = np.column_stack([
        np.full(len(proxy_z_centers), well_x_m),
        np.full(len(proxy_z_centers), well_y_m),
        proxy_z_centers
    ])
    
    # Interpolate
    proxy_rh = rh_interpolator(query_points)
    proxy_rv = rv_interpolator(query_points)
    
    return proxy_rh, proxy_rv


def simulate_logging_point(model, logging_point_idx, data_type='UDAR'):
    """
    Simulate NN response for a single logging point.
    
    Parameters:
    -----------
    model : dict
        Dictionary with 'formation' and 'wellpath' keys
    logging_point_idx : int
        Index of the logging point in wellpath
    data_type : str
        'UDAR' or 'Bfield'
    
    Returns:
    --------
    nn_pred : np.ndarray
        Predictions from neural network
    proxy_rh : np.ndarray
        Rh values on proxy grid
    proxy_rv : np.ndarray
        Rv values on proxy grid
    """
    wellpath = model['wellpath']
    
    # Get well position at this logging point
    well_x_m = wellpath['X'][logging_point_idx][0]
    well_y_m = wellpath['Y'][logging_point_idx][0]
    well_z_m = wellpath['Z'][logging_point_idx][0]
    
   # print(f"Logging point {logging_point_idx}: X={well_x_m:.2f}, Y={well_y_m:.2f}, Z={well_z_m:.2f} m")
    
    # Create proxy grid
    proxy_z_centers = create_proxy_grid(well_z_m, grid_size, cell_thickness_m, well_cell_index)
    
    # Interpolate formation model onto proxy grid
    proxy_rh, proxy_rv = interpolate_formation_to_proxy(
        model, well_x_m, well_y_m, well_z_m,
        proxy_z_centers, out_of_region_value
    )
    
    # Build resistivity tensor: shape (1, 3, grid_size)
    resistivity = np.zeros((1, 3, grid_size))
    resistivity[0, 0, :] = proxy_rh  # Rh
    resistivity[0, 1, :] = proxy_rv  # Rv
    resistivity[0, 2, well_cell_index] = 1.0  # Well position one-hot
    
    # Convert to tensor and run forward pass
    resistivity_tensor = torch.tensor(resistivity, dtype=torch.float32, device=device)
    nn_pred_tensor = NN_sim.image_to_log(resistivity_tensor)
    
    # Convert NN bfield predictions to UDAR if needed
    if data_type == 'UDAR':
        nn_pred = convert_bfield_to_udar(nn_pred_tensor[:, :, :10].detach().cpu().numpy())
    else:
        nn_pred = nn_pred_tensor[:, :, :10].detach().cpu().numpy()

    
    # Simulate 1D response
    resistivity = {'rh': np.log(np.concatenate([np.array([1]), proxy_rh])),
                    'rv': np.log(np.concatenate([np.array([1]), proxy_rv]))}
    UTA1D_sim.tool['tvd'] = TVD[logging_point_idx:logging_point_idx+1]
    UTA1D_sim.tool['surface_depth'] = [TVD[logging_point_idx]+(cell_thickness_ft/2) - cell_thickness_ft*np.ceil(grid_size/2) + cell_thickness_ft*i for i in range(grid_size+1)]
    oneD_pred = UTA1D_sim.run_fwd_sim(resistivity, 0)


    # Simulate 2D response
    UTA2D_sim.model['shift'] = {'x':well_x_m*3.28084 - UTA2D_input['dX']*np.ceil(UTA2D_input['dims'][0]/2),
                            'y':0,
                            'z':UTA1D_sim.tool['surface_depth'][0]}
    UTA2D_sim.tool['MD'] = MD[logging_point_idx:logging_point_idx+1]
    twoD_pred = UTA2D_sim.run_fwd_sim({'rh': np.log(proxy_rh),
                                       'rv': np.log(proxy_rv)},
                                       0)



    return nn_pred[0], copy.deepcopy(oneD_pred), copy.deepcopy(twoD_pred)


def simulate_all_logging_points(model, data_type='UDAR'):
    """
    Simulate NN response for all logging points in the wellpath.
    
    Parameters:
    -----------
    model : dict
        Dictionary with 'formation' and 'wellpath' keys
    data_type : str
        'UDAR' or 'Bfield'
    
    Returns:
    --------
    proxy_predictions : list
        List of predictions for each logging point from NN-proxy
    oneD_predictions : list
        List of predictions for each logging point from 1D simulation
    """
    wellpath = model['wellpath']
    n_logging_points = len(wellpath['Z'])
    
    proxy_predictions = []
    oneD_predictions = []
    twoD_predictions = []
    
    for idx in range(0,n_logging_points,1):
        nn_pred, oneD_pred, twoD_pred = simulate_logging_point(model, idx, data_type)
        proxy_predictions.append(nn_pred)
        oneD_predictions.append(oneD_pred)
        twoD_predictions.append(twoD_pred)

        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{n_logging_points} logging points")
    
    return proxy_predictions, oneD_predictions, twoD_predictions


# Example usage
if __name__ == "__main__":
    path_to_model = "/home/AD.NORCERESEARCH.NO/krfo/CodeProjects/DISTINGUISH/Jacobian/inversion/data/Benchmark-3/"

    with h5py.File(path_to_model + "globalmodel.h5", "r") as f:
        model = h5_to_dict(f)
    
    nosim = True  # Set to True to load saved results instead of simulating
    if nosim:
        # Load previously saved results
        loaded = np.load('simulation_results.npz', allow_pickle=True)
        nn_predictions = loaded['nn_predictions']
        oneD_predictions = loaded['oneD_predictions']
        twoD_predictions = loaded['twoD_predictions']
    else:
        # Run simulations
        nn_predictions, oneD_predictions, twoD_predictions = simulate_all_logging_points(model, data_type='UDAR')
        
        # Save results
        np.savez_compressed('simulation_results.npz', 
                            nn_predictions=np.array(nn_predictions),
                            oneD_predictions=oneD_predictions,
                            twoD_predictions=twoD_predictions)
    
    # import the measured data for comparison
    data = pd.read_pickle('../EnRML/data.pkl')

    # Plot results
    import matplotlib.pyplot as plt
    
    # Tool configurations
    tools = [('6kHz','83ft'), ('12kHz','83ft'), ('24kHz','83ft'), 
             ('24kHz','43ft'), ('48kHz','43ft'), ('96kHz','43ft')]
    tool_labels = [f"{freq}, {spacing}" for freq, spacing in tools]
    tool_keys = [f"('{freq}','{spacing}')" for freq, spacing in tools]
    
    # Measurement names for NN predictions (order in array)
    measurement_names = ['USDP', 'USDA', 'UADP', 'UADA', 'UHRP', 'UHRA', 'UHAP', 'UHAA']
    
    # Measurement names for 1D predictions (order in dict values)
    oneD_measurement_names = ['USDA', 'USDP', 'UADA', 'UADP', 'UHRA', 'UHRP', 'UHAA', 'UHAP']
    
    # Panel groupings: (measurement indices, panel title)
    panels = [
        ([0, 1], 'USD - Phase and Attenuation'),
        ([2, 3], 'UAD - Phase and Attenuation'),
        ([4, 5], 'UHR - Phase and Attenuation'),
        ([6, 7], 'UHA - Phase and Attenuation')
    ]
    
    # Convert NN predictions to numpy array: shape (n_logging_points, 6, 8)
    nn_predictions_array = np.array(nn_predictions)
    n_logging_points = nn_predictions_array.shape[0]
    
    # Get depth/distance from wellpath
    distances = model['wellpath']['X'][:, 0]  # Meters drilled

    # Extract measured data in same format as predictions: shape (n_logging_points, 6, 8)
    measured_data_array = np.zeros((n_logging_points, 6, 8))
    
    for el in range(n_logging_points):  # logging point index
        for tool_idx, dat in enumerate(tools):  # tool index
            if tool_idx < 6:  # Only use first 6 tools to match array shape
                # Extract measurements for this tool
                tool_measurements = data.iloc[el][dat][[measurement_names.index(val) for val in measurement_names]]
                measured_data_array[el, tool_idx, :] = tool_measurements
    
    # ========== Plot NN Proxy Predictions ==========
    
    # Color map for tools
    colors = plt.cm.tab10(np.linspace(0, 0.6, 6))

    num_of_tools = range(6)  # First 6 tools
    plot_phase = 'Atten'  # 'Phase' or 'Atten'
    
    for tool_idx in num_of_tools:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('NN Proxy Predictions', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        for panel_idx, (meas_indices, panel_title) in enumerate(panels):
            ax = axes[panel_idx]
            
            for meas_idx in meas_indices:
                # Extract data: shape (n_logging_points,)
                data = nn_predictions_array[:, tool_idx, meas_idx]
                
                # Line style: solid for phase (even indices), dashed for attenuation (odd indices)
                linestyle = '-' if meas_idx % 2 == 0 else '--'
                
                # Label includes tool and measurement type
                meas_type = 'Phase' if meas_idx % 2 == 0 else 'Atten'
                label = f"{tool_labels[tool_idx]} ({meas_type})"

                if meas_type != plot_phase:
                    continue
                
                ax.plot(distances, data, linestyle=linestyle, color=colors[tool_idx], 
                        label=label, linewidth=1.5)
                
                # Plot measured data (markers)
                measured_data = measured_data_array[:, tool_idx, meas_idx]
                ax.scatter(distances, measured_data, marker='o', color=colors[tool_idx], 
                            s=15, label=f"{tool_labels[tool_idx]} Measured ({meas_type})")
            
            ax.set_xlabel('Distance (m)', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.set_title(panel_title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, ncol=2, loc='best')
        
        plt.tight_layout()
        plt.savefig(f'nn_proxy_predictions_tool_{tool_idx}.png', dpi=300, bbox_inches='tight')
        print(f"NN Proxy plot saved as nn_proxy_predictions_tool_{tool_idx}.png")
        plt.close(fig)
    
    # ========== Plot 1D Simulator Predictions ==========
    # Extract 1D predictions into array format: shape (n_logging_points, 6, 8)
    oneD_predictions_array = np.zeros((n_logging_points, 6, 8))
    
    for log_idx in range(n_logging_points):
        pred_dict = oneD_predictions[log_idx][0]  # First element is the dictionary
        for tool_idx, tool_key in enumerate(tool_keys):
            if tool_key in pred_dict:
                oneD_predictions_array[log_idx, tool_idx, :] = pred_dict[tool_key]


    # G(1)=USDA
    # G(2)=USDP
    # G(3)=UADA
    # G(4)=UADP
    # G(5)=UHRA
    # G(6)=UHRP
    # G(7)=UHAA
    # G(8)=UHAP
    
    # Map 1D measurement order to NN measurement order
    # 1D order: USDA, USDP, UADA, UADP, UHRA, UHRP, UHAA, UHAP (indices 0-7)
    # NN order: USDP, USDA, UADP, UADA, UHRP, UHRA, UHAP, UHAA (indices 0-7)
    # Mapping: 1D idx -> NN idx: {0->1, 1->0, 2->3, 3->2, 4->5, 5->4, 6->7, 7->6}
    #oneD_to_nn_mapping = [1, 0, 3, 2, 5, 4, 7, 6]
    oneD_to_nn_mapping = [0,1,2,3,4,5,6,7]  # No reordering for current use case

    oneD_predictions_reordered = oneD_predictions_array[:, :, oneD_to_nn_mapping]
    
    for tool_idx in num_of_tools:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('1D Simulator Predictions', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        for panel_idx, (meas_indices, panel_title) in enumerate(panels):
            ax = axes[panel_idx]
            
            for meas_idx in meas_indices:
                # Extract data: shape (n_logging_points,)
                data = oneD_predictions_reordered[:, tool_idx, meas_idx]
                
                # Line style: solid for phase (even indices), dashed for attenuation (odd indices)
                linestyle = '-' if meas_idx % 2 == 0 else '--'
                
                # Label includes tool and measurement type
                meas_type = 'Phase' if meas_idx % 2 == 0 else 'Atten'
                label = f"{tool_labels[tool_idx]} ({meas_type})"

                if meas_type != plot_phase:
                    continue
                
                ax.plot(distances, data, linestyle=linestyle, color=colors[tool_idx], 
                    label=label, linewidth=1.5)
                
                # Plot measured data (markers)
                measured_data = measured_data_array[:, tool_idx, meas_idx]
                ax.scatter(distances, measured_data, marker='o', color=colors[tool_idx], 
                        s=15, label=f"{tool_labels[tool_idx]} Measured ({meas_type})")
                    
            ax.set_xlabel('Distance (m)', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.set_title(panel_title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, ncol=2, loc='best')
        
        plt.tight_layout()
        plt.savefig(f'oneD_simulator_predictions_{tool_idx}.png', dpi=300, bbox_inches='tight')
        print(f"1D Simulator plot saved as oneD_simulator_predictions_{tool_idx}.png")
        plt.close(fig)
        
    
    # ========== Plot 2D Simulator Predictions ==========
    # Extract 2D predictions into array format: shape (n_logging_points, 6, 8)
    twoD_predictions_array = np.zeros((n_logging_points, 6, 8))
    
    for log_idx in range(n_logging_points):
        pred_dict = twoD_predictions[log_idx][0]  # First element is the dictionary
        for tool_idx, tool_key in enumerate(tool_keys):
            if tool_key in pred_dict:
                twoD_predictions_array[log_idx, tool_idx, :] = pred_dict[tool_key]
    
    # Map 2D measurement order to NN measurement order (same as 1D)
    twoD_to_nn_mapping = [0,1,2,3,4,5,6,7]  # No reordering for current use case
    twoD_predictions_reordered = twoD_predictions_array[:, :, twoD_to_nn_mapping]
    
    
    for tool_idx in num_of_tools:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('2D Simulator Predictions', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        for panel_idx, (meas_indices, panel_title) in enumerate(panels):
            ax = axes[panel_idx]
            for meas_idx in meas_indices:
                # Extract data: shape (n_logging_points,)
                data = twoD_predictions_reordered[:, tool_idx, meas_idx]
                
                # Line style: solid for phase (even indices), dashed for attenuation (odd indices)
                linestyle = '-' if meas_idx % 2 == 0 else '--'
                
                # Label includes tool and measurement type
                meas_type = 'Phase' if meas_idx % 2 == 0 else 'Atten'
                label = f"{tool_labels[tool_idx]} ({meas_type})"

                if meas_type != plot_phase:
                    continue
                
                ax.plot(distances, data, linestyle=linestyle, color=colors[tool_idx], 
                    label=label, linewidth=1.5)
                
                # Plot measured data (markers)
                measured_data = measured_data_array[:, tool_idx, meas_idx]
                ax.scatter(distances, measured_data, marker='o', color=colors[tool_idx], 
                        s=15, label=f"{tool_labels[tool_idx]} Measured ({meas_type})")
                    
            ax.set_xlabel('Distance (m)', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.set_title(panel_title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, ncol=2, loc='best')
        
        plt.tight_layout()
        plt.savefig(f'twoD_simulator_predictions_{tool_idx}.png', dpi=300, bbox_inches='tight')
        print(f"2D Simulator plot saved as twoD_simulator_predictions_{tool_idx}.png")
        plt.close(fig)
       # plt.show()
    
    # ========== Plot Comparison (NN vs 1D) ==========
   
    for tool_idx in num_of_tools:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Comparison: NN Proxy (solid) vs 1D Simulator (dashed)', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        for panel_idx, (meas_indices, panel_title) in enumerate(panels):
            ax = axes[panel_idx]
            for meas_idx in meas_indices:
                # NN data
                nn_data = nn_predictions_array[:, tool_idx, meas_idx]
                # 1D data
                oneD_data = oneD_predictions_reordered[:, tool_idx, meas_idx]
                
                # Label
                meas_type = 'Phase' if meas_idx % 2 == 0 else 'Atten'

                # only plot 'Phase' measurements
                if meas_type != plot_phase:
                    continue

                # Plot NN (solid line)
                ax.plot(distances, nn_data, linestyle='-', color=colors[tool_idx], 
                    linewidth=1.5, alpha=0.7)
                
                # Plot 1D (dashed line)
                ax.plot(distances, oneD_data, linestyle='--', color=colors[tool_idx], 
                    linewidth=1.5, alpha=0.7, label=f"{tool_labels[tool_idx]} ({meas_type})")
                

                # Plot measured data (markers)
                measured_data = measured_data_array[:, tool_idx, meas_idx]
                ax.scatter(distances, measured_data, marker='o', color=colors[tool_idx], 
                        s=15, label=f"{tool_labels[tool_idx]} Measured ({meas_type})")
            
            ax.set_xlabel('Distance (m)', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.set_title(panel_title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, ncol=2, loc='best')
        
        plt.tight_layout()
        plt.savefig(f'comparison_nn_vs_1d_{tool_idx}.png', dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved as comparison_nn_vs_1d_{tool_idx}.png")
        plt.close(fig)
        #plt.show()

