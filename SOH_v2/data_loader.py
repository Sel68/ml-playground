import os
import glob
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.stats import kurtosis, skew

class BatterySOHDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

def extract_temperature(filename):
    # E.g. 'CY25-025_1-#1.csv' -> 25
    # CYX-Y_Z-#N
    match = re.search(r'CY(\d+)', filename)
    if match:
        return float(match.group(1))
    return 25.0  # Default if not found

def compute_features(df_cycle):
    # 4 variables: Voltage, Current, Discharge Capacity, Charge Capacity
    # If a column doesn't exist or is empty, we handle it safely.
    v = df_cycle['Ecell/V'].values
    i = df_cycle['<I>/mA'].values
    q_dis = df_cycle['Q discharge/mA.h'].values
    q_chg = df_cycle['Q charge/mA.h'].values
    
    features = []
    for sig in [v, i, q_dis, q_chg]:
        if len(sig) > 0:
            features.extend([np.mean(sig), np.std(sig), skew(sig), kurtosis(sig)])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
            
    # Relaxation features (Voltage during I approx 0)
    # We define relaxation as current magnitude < 10 mA
    relax_idx = np.abs(i) < 10.0
    v_relax = v[relax_idx]
    if len(v_relax) > 0:
        features.extend([np.var(v_relax), skew(v_relax), np.max(v_relax)])
    else:
        features.extend([0.0, 0.0, 0.0])
        
    return np.nan_to_num(np.array(features))

def prepare_dataset(data_dir, test_ratio=0.2, val_ratio=0.2):
    all_files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    # We split by cell ID. In this dataset, a file is a cell.
    np.random.seed(42)
    np.random.shuffle(all_files)
    
    n_files = len(all_files)
    n_test = int(n_files * test_ratio)
    n_val = int(n_files * val_ratio)
    n_train = n_files - n_test - n_val
    
    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_train+n_val]
    test_files = all_files[n_train+n_val:]
    
    def process_files(files):
        data_list = []
        for file in files:
            temp = extract_temperature(os.path.basename(file))
            df = pd.read_csv(file)
            
            cycles = df['cycle number'].unique()
            for cycle_idx, cycle_num in enumerate(sorted(cycles)):
                # We need features, SOH, temperature, and cycle count
                df_c = df[df['cycle number'] == cycle_num]
                soh = df_c['Q discharge/mA.h'].max()
                
                # SOH > 0 filter (sometimes erroneous cycles have 0 capacity)
                if soh > 10.0:
                    features = compute_features(df_c)
                    data_list.append({
                        'features': features,
                        'soh': soh,
                        'temperature': temp,
                        'cycle': cycle_idx + 1 # Use sequential index as delta t is 1
                    })
        return data_list

    print("Processing training data...")
    train_data = process_files(train_files)
    print("Processing validation data...")
    val_data = process_files(val_files)
    print("Processing testing data...")
    test_data = process_files(test_files)
    
    # Min-max normalization for features
    all_features = np.array([d['features'] for d in train_data])
    f_min = all_features.min(axis=0, keepdims=True)
    f_max = all_features.max(axis=0, keepdims=True)
    
    # Prevent division by zero
    f_range = f_max - f_min
    f_range[f_range == 0] = 1e-6
    
    # Normalize SOH (Target variable)
    all_soh = np.array([d['soh'] for d in train_data])
    soh_min = all_soh.min()
    soh_max = all_soh.max()
    soh_range = soh_max - soh_min
    if soh_range == 0: soh_range = 1e-6
    
    def normalize_list(d_list):
        for d in d_list:
            feat = d['features']
            feat_norm = 2 * ((feat - f_min) / f_range) - 1.0
            d['features'] = feat_norm.squeeze()
            
            d['soh_norm'] = (d['soh'] - soh_min) / soh_range
        return d_list
        
    train_data = normalize_list(train_data)
    val_data = normalize_list(val_data)
    test_data = normalize_list(test_data)
    
    return train_data, val_data, test_data, train_files, val_files, test_files, f_min, f_range, soh_min, soh_range

def get_dataloaders(data_dir, batch_size=64):
    train_data, val_data, test_data, train_files, val_files, test_files, f_min, f_range, soh_min, soh_range = prepare_dataset(data_dir)
    
    def collate_fn(batch):
        features = torch.tensor(np.array([item['features'] for item in batch]), dtype=torch.float32)
        soh = torch.tensor(np.array([item['soh_norm'] for item in batch]), dtype=torch.float32).unsqueeze(1)
        temp = torch.tensor(np.array([item['temperature'] for item in batch]), dtype=torch.float32).unsqueeze(1)
        cycle = torch.tensor(np.array([item['cycle'] for item in batch]), dtype=torch.float32).unsqueeze(1)
        return features, soh, temp, cycle

    train_loader = DataLoader(BatterySOHDataset(train_data), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(BatterySOHDataset(val_data), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(BatterySOHDataset(test_data), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader
