import torch
import torch.nn as nn
import numpy as np

def compute_loss(model, inputs, targets, lambda_mono=0.1):
    """
    Total Loss = Data Loss + lambda_mono * Monotonicity Loss
    """
    # Prediction
    soh_pred = model(inputs)
    
    # 1. Data Loss
    loss_data = nn.MSELoss()(soh_pred, targets)
    
    # 2. Monotonicity Loss
    # We expect SOH[n+1] <= SOH[n]
    # L_mono = mean(ReLU(soh_pred[n+1] - soh_pred[n]))
    # Note: This requires inputs to be sorted by cycle for each battery
    loss_mono = torch.tensor(0.0, device=inputs.device)
    
    # Assuming the batch might contain multiple batteries, we should ideally
    # split by battery. But if we train in full sequences, we can just diff.
    # For now, let's assume inputs are somewhat ordered or we process per battery in the loop.
    
    # Simple diff approach for sequential data:
    diff = soh_pred[1:] - soh_pred[:-1]
    loss_mono = torch.mean(torch.relu(diff)) # Penalize increases
    
    total_loss = loss_data + lambda_mono * loss_mono
    
    return total_loss, loss_data, loss_mono
