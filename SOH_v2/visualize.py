import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import prepare_dataset, extract_temperature, compute_features
from model import BatteryPINN

def visualize(data_dir):
    print("Preparing dataset (loading files)...")
    train_data, val_data, test_data, train_files, val_files, test_files, f_min, f_range, soh_min, soh_range = prepare_dataset(data_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading model...")
    model = BatteryPINN(n_features=19).to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()
    
    os.makedirs("plots", exist_ok=True)
    
    # 1. SOH Trajectories
    print("Plotting SOH trajectories for test cells...")
    temp_batches = {}  # For t_arr plot
    
    for file in test_files:
        temp = extract_temperature(os.path.basename(file))
        df = pd.read_csv(file)
        
        cycles = df['cycle number'].unique()
        cycles = sorted(cycles)
        
        cell_cycles = []
        cell_soh_true = []
        cell_soh_pred = []
        cell_t_arr = []
        
        for cycle_idx, cycle_num in enumerate(cycles):
            df_c = df[df['cycle number'] == cycle_num]
            soh = df_c['Q discharge/mA.h'].max()
            
            if soh > 10.0:
                features = compute_features(df_c)
                # Normalize
                feat_norm = 2 * ((features - f_min) / f_range) - 1.0
                
                # To tensor
                x_t = torch.tensor(feat_norm, dtype=torch.float32).to(device)
                temp_t = torch.tensor([[temp]], dtype=torch.float32).to(device)
                cycle_t = torch.tensor([[cycle_idx + 1]], dtype=torch.float32).to(device)
                
                with torch.no_grad():
                    u_pred, t_arr = model(x_t, temp_t, cycle_t)
                    
                u_pred_val = u_pred.item() * soh_range + soh_min
                
                cell_cycles.append(cycle_idx + 1)
                cell_soh_true.append(soh)
                cell_soh_pred.append(u_pred_val)
                cell_t_arr.append(t_arr.item())
        
        if temp not in temp_batches:
            temp_batches[temp] = []
        temp_batches[temp].append((cell_cycles, cell_t_arr))
        
        # Plot SOH
        plt.figure(figsize=(8, 5))
        plt.plot(cell_cycles, cell_soh_true, label='True SOH', color='black', linewidth=2)
        plt.plot(cell_cycles, cell_soh_pred, label='Predicted SOH', color='red', linestyle='dashed')
        plt.xlabel('Cycle')
        plt.ylabel('SOH (Max Discharge Capacity / mA.h)')
        plt.title(f'SOH Trajectory - {os.path.basename(file)} (Temp: {temp}°C)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"plots/soh_{os.path.basename(file)}.png")
        plt.close()
        
    # 2. Thermal Age vs Cycle Number
    print("Plotting Thermal Age (t_arr) vs Cycle Number...")
    plt.figure(figsize=(8, 5))
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    c_idx = 0
    for temp, lst in temp_batches.items():
        # Just plot the first cell of this temperature as representative
        if len(lst) > 0:
            c, t_arr = lst[0]
            plt.plot(c, t_arr, label=f'{temp}°C', color=colors[c_idx % len(colors)], linewidth=2)
            c_idx += 1
            
    plt.xlabel('Cycle Number (t)')
    plt.ylabel('Equivalent Aging Time (t_arr)')
    plt.title('Thermal Age vs Cycle Number across Temperatures')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/thermal_age_vs_cycle.png')
    plt.close()

if __name__ == '__main__':
    if not os.path.exists('best_model.pth'):
        print("Error: best_model.pth not found. Please run train.py first.")
    else:
        visualize('Dataset_1_NCA_battery')
