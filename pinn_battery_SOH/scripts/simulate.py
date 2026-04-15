import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from model import BatteryNaturePINN

def simulate_climates(model_path, stats_path, climate_path, cycle_limit=500):
    device = torch.device('cpu')
    with open(stats_path, 'r') as f:
        stats = json.load(f)
        
    model = BatteryNaturePINN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    climate_df = pd.read_csv(climate_path)
    
    # We need to simulate the evolution of features.
    # In a real scenario, features change with temperature.
    # For this simulation, we'll assume a simplified impact:
    # Temperature speeds up cycle-based degradation.
    
    # However, the model takes features from charging cycles.
    # We'll use a reference cycle's features and modify them based on Temperature.
    # Or, as a simplification requested by the user, we'll simulate the SOH trajectory.
    
    results = {}
    
    # Reference features (average from B0005)
    ref_feats = np.zeros(16) # Placeholder
    
    # Since we don't have a generative model for features, we'll simulate by 
    # adjusting the "cycle" input to reflect "thermal aging".
    # Equivalent Cycles = Cycles * exp(-Ea / R * (1/T - 1/Tref))
    
    for region in ['Anchorage_Tavg', 'Kuwait_Tavg']:
        sohs = []
        # Mean annual temp
        T_avg = climate_df[region].mean() # in Celsius
        T_ref = 25.0
        
        # Acceleration factor (simple Arrhenius)
        # Factor > 1 for hot, < 1 for cold
        # k = exp(-Ea/R * (1/T - 1/Tref))
        Ea = 30000 
        R = 8.314
        k = np.exp(-(Ea/R) * (1/(T_avg + 273.15) - 1/(T_ref + 273.15)))
        
        # Fetch a reasonable starting feature vector from the dataset
        # (normalized to the stats)
        
        for n in range(1, cycle_limit + 1):
            # Effective cycle
            eff_n = n * k
            
            # Construct input vector (16 features fixed, 1 cycle varying)
            # To show contrast, we'll use a "representative" feature vector
            x_raw = np.zeros(17)
            x_raw[16] = eff_n # cycle index
            
            # Normalize cycle
            cycle_min = stats['cycle']['min']
            cycle_max = stats['cycle']['max']
            n_norm = 2.0 * (eff_n - cycle_min) / (cycle_max - cycle_min + 1e-9) - 1.0
            
            # For simplicity in simulation of the *model itself*, we'll use the cycle-induced fade
            # But we must normalize all features. We'll use 0.0 for others (mean).
            x_norm = np.zeros((1, 17))
            x_norm[0, 16] = n_norm
            
            with torch.no_grad():
                soh = model(torch.tensor(x_norm, dtype=torch.float32)).item()
            sohs.append(soh)
        results[region] = sohs

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), gridspec_kw={'height_ratios': [2, 1]})
    
    # SOH Plot
    ax1.plot(range(1, cycle_limit + 1), results['Anchorage_Tavg'], label='Anchorage (Cold)', color='#3498db', linewidth=2)
    ax1.plot(range(1, cycle_limit + 1), results['Kuwait_Tavg'], label='Kuwait (Hot)', color='#e74c3c', linewidth=2)
    ax1.axhline(y=0.7, color='#7f8c8d', linestyle='--', label='End of Life (70%)')
    ax1.set_title('Battery SOH Degradation: Cold vs. Hot Climate', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Cycle Number', fontsize=12)
    ax1.set_ylabel('State of Health (SOH)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.65, 1.05)
    
    # Temperature Profile Plot
    months = np.arange(1, 13)
    ax2.plot(months, climate_df['Anchorage_Tavg'], marker='o', label='Anchorage Tavg', color='#3498db')
    ax2.plot(months, climate_df['Kuwait_Tavg'], marker='o', label='Kuwait Tavg', color='#e74c3c')
    ax2.set_xticks(months)
    ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax2.set_title('Annual Temperature Profiles', fontsize=12)
    ax2.set_xlabel('Month', fontsize=12)
    ax2.set_ylabel('Temperature (°C)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sim_results.png', dpi=300)
    plt.show()
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default='checkpoints/nature_best_model.pth')
    parser.add_argument("--stats_path", default='checkpoints/nature_stats.json')
    parser.add_argument("--climate_path", default='climate_profiles.csv')
    parser.add_argument("--cycle_limit", type=int, default=500)
    args = parser.parse_args()
    
    simulate_climates(args.model_path, args.stats_path, args.climate_path, args.cycle_limit)
