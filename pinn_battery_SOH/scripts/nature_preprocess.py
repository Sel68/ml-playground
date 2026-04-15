import os
import datetime
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, entropy

def parse_matlab_datevec(s):
    try:
        s = s.strip().strip('[]')
        vals = s.split()
        if len(vals) < 6:
            vals = [v.strip() for v in s.replace(',', ' ').split()]
            if len(vals) < 6:
                return None
        vals = [float(v) for v in vals]
        year, month, day, hour, minute, sec = vals[:6]
        dt = datetime.datetime(int(year), int(month), int(day),
                               int(hour), int(minute), int(sec))
        return dt
    except:
        return None

def extract_nature_features(V, I, T):
    """
    Extracts 16 features as per Nature 2024 paper.
    Voltage range: [Vend-0.2, Vend]
    Current range during CV: [0.1, 0.5]
    Since we don't have Vend explicitly, we assume Vend is the max voltage in the cycle.
    """
    Vend = np.max(V)
    
    # Feature 1-8 from Voltage in [Vend-0.2, Vend]
    v_mask = (V >= Vend - 0.2) & (V <= Vend)
    V_sub = V[v_mask]
    I_sub_v = I[v_mask]
    T_sub_v = T[v_mask]
    
    # Feature 9-16 from Current in [0.1, 0.5]
    # This usually corresponds to the CV phase
    i_mask = (I <= 0.5) & (I >= 0.1)
    I_sub_i = I[i_mask]
    V_sub_i = V[i_mask]
    T_sub_i = T[i_mask]
    
    features = {}
    
    def get_stats(data, time, prefix):
        if len(data) < 2:
            return [0.0]*8
        dt = np.diff(time)
        acc_q = np.sum(np.abs(data[:-1]) * dt) / 3600.0 # Ah
        slope = (data[-1] - data[0]) / (time[-1] - time[0] + 1e-6)
        
        # Entropy - simple discretization
        hist, _ = np.histogram(data, bins=10, density=True)
        ent = entropy(hist + 1e-9)
        
        return [
            np.mean(data),
            np.std(data),
            kurtosis(data),
            skew(data),
            time[-1] - time[0],
            acc_q,
            slope,
            ent
        ]
    
    f_v = get_stats(V_sub, T_sub_v, "V")
    f_i = get_stats(I_sub_i, T_sub_i, "I")
    
    return f_v + f_i

def load_nature_dataset(metadata_path, data_dir, Q_rated=2.0):
    df_meta = pd.read_csv(metadata_path)
    df_meta['start_time_parsed'] = df_meta['start_time'].apply(parse_matlab_datevec)
    df_meta = df_meta.dropna(subset=['start_time_parsed'])
    df_meta = df_meta.sort_values(by=['battery_id', 'start_time_parsed']).reset_index(drop=True)
    
    # Numbering cycles
    df_meta['cycle_num'] = df_meta.groupby('battery_id').apply(
        lambda g: (g['type'] == 'discharge').cumsum()
    ).reset_index(level=0, drop=True)
    
    # We need to link charge features to the capacity measured in the next discharge cycle
    # or the capacity found in metadata.
    
    processed_rows = []
    
    # Group by battery_id
    for bid, group in df_meta.groupby('battery_id'):
        print(f"Processing battery {bid}...")
        group = group.reset_index(drop=True)
        
        for i in range(len(group) - 1):
            row = group.iloc[i]
            if row['type'] == 'charge':
                # Look for the next discharge to get Capacity
                next_discharge = group.iloc[i+1:]
                next_discharge = next_discharge[next_discharge['type'] == 'discharge']
                if next_discharge.empty:
                    continue
                
                capacity = pd.to_numeric(next_discharge.iloc[0]['Capacity'], errors='coerce')
                if pd.isna(capacity) or capacity <= 0:
                    continue
                
                # Load charge data
                filename = str(row['filename']).strip()
                path = os.path.join(data_dir, filename)
                if not os.path.exists(path): continue
                
                try:
                    df_ts = pd.read_csv(path)
                    df_ts.columns = df_ts.columns.str.strip()
                    V = df_ts['Voltage_measured'].values
                    I = df_ts['Current_measured'].values
                    T = df_ts['Time'].values
                    
                    feats = extract_nature_features(V, I, T)
                    
                    processed_rows.append({
                        'battery_id': bid,
                        'cycle': int(row['cycle_num']),
                        'SOH': capacity / Q_rated,
                        'capacity': capacity,
                        **{f'f{idx}': val for idx, val in enumerate(feats)}
                    })
                except:
                    continue
                    
    df_res = pd.DataFrame(processed_rows)
    df_res.to_csv('nature_processed.csv', index=False)
    print(f"Saved {len(df_res)} rows to nature_processed.csv")
    return df_res

if __name__ == "__main__":
    load_nature_dataset(
        "/mnt/disk4/GitHub/Battery-degration-pinn/cleaned_dataset/metadata.csv",
        "/mnt/disk4/GitHub/Battery-degration-pinn/cleaned_dataset/data"
    )
