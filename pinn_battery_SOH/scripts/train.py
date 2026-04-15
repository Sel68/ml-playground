import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import json
from model import BatteryNaturePINN
from loss import compute_loss

def train_nature_pinn(data_path, epochs=100, lr=1e-3, lambda_mono=0.1, hidden_dim=128, checkpoint_dir='checkpoints'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    df = pd.read_csv(data_path)
    
    # Simple split: B0005, B0006 for train, B0007 for val
    train_ids = ['B0005', 'B0006', 'B0025', 'B0026', 'B0027', 'B0028'] # Using more for diversity
    val_ids = ['B0007', 'B0018']
    
    df_train = df[df['battery_id'].isin(train_ids)].sort_values(['battery_id', 'cycle']).copy()
    df_val = df[df['battery_id'].isin(val_ids)].sort_values(['battery_id', 'cycle']).copy()
    
    # Normalization (Min-Max as per paper)
    feat_cols = [f'f{i}' for i in range(16)] + ['cycle']
    
    stats = {}
    for col in feat_cols:
        stats[col] = {
            'min': float(df_train[col].min()),
            'max': float(df_train[col].max())
        }
        
    with open(os.path.join(checkpoint_dir, 'nature_stats.json'), 'w') as f:
        json.dump(stats, f)
        
    def apply_norm(d, s):
        for col in feat_cols:
            denom = s[col]['max'] - s[col]['min'] + 1e-9
            d[col + '_norm'] = 2.0 * (d[col] - s[col]['min']) / denom - 1.0
        return d
        
    df_train = apply_norm(df_train, stats)
    df_val = apply_norm(df_val, stats)
    
    feat_norm_cols = [c + '_norm' for c in feat_cols]
    
    train_X = torch.tensor(df_train[feat_norm_cols].values, dtype=torch.float32)
    train_y = torch.tensor(df_train['SOH'].values, dtype=torch.float32).view(-1, 1)
    
    val_X = torch.tensor(df_val[feat_norm_cols].values, dtype=torch.float32)
    val_y = torch.tensor(df_val['SOH'].values, dtype=torch.float32).view(-1, 1)
    
    model = BatteryNaturePINN(input_dim=17, hidden_dim=hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        t_loss, t_data, t_mono = compute_loss(model, train_X, train_y, lambda_mono)
        t_loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            v_loss, v_data, v_mono = compute_loss(model, val_X, val_y, lambda_mono)
            
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | Loss: {t_loss.item():.6f} | Val Data Loss: {v_data.item():.6f} | Val Mono: {v_mono.item():.6f}")
            
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'nature_best_model.pth'))
            
    print(f"Training Complete. Best Val Loss: {best_val_loss.item():.6f}")
    return model

if __name__ == "__main__":
    if os.path.exists("nature_processed.csv"):
        train_nature_pinn("nature_processed.csv", epochs=100)
