import torch
import numpy as np
from data_loader import get_dataloaders
from model import BatteryPINN, compute_pinn_losses

def train_model(data_dir, epochs=200, batch_size=128, patience=20):
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(data_dir, batch_size=batch_size)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Initializing model...")
    model = BatteryPINN(n_features=19).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        train_pde = 0.0
        train_mono = 0.0
        
        for batch_i, (features, soh, temp, cycle) in enumerate(train_loader):
            features, soh, temp, cycle = features.to(device), soh.to(device), temp.to(device), cycle.to(device)
            optimizer.zero_grad()
            
            loss, mse_loss, pde_loss, mono_loss = compute_pinn_losses(model, features, temp, cycle, soh, alpha=0.1, beta=0.1)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * features.size(0)
            train_mse += mse_loss.item() * features.size(0)
            train_pde += pde_loss.item() * features.size(0)
            train_mono += mono_loss.item() * features.size(0)
            
        n_train = len(train_loader.dataset)
        train_loss /= n_train
        train_mse /= n_train
        train_pde /= n_train
        train_mono /= n_train
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for features, soh, temp, cycle in val_loader:
                pass
        
        # We will compute full PINN validation loss using enable_grad()
        with torch.enable_grad():
            for features, soh, temp, cycle in val_loader:
                features, soh, temp, cycle = features.to(device), soh.to(device), temp.to(device), cycle.to(device)
                loss, _, _, _ = compute_pinn_losses(model, features, temp, cycle, soh, alpha=0.1, beta=0.1)
                val_loss += loss.item() * features.size(0)
        
        n_val = len(val_loader.dataset)
        val_loss /= n_val
        
        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.6f} [MSE:{train_mse:.6f} PDE:{train_pde:.6f} MONO:{train_mono:.6f}] | Val Loss: {val_loss:.6f} | Ea = {model.Ea.item():.2f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print("  --> Saved new best model")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
                
    print("Training finished. Evaluating on test set...")
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    test_mse = 0.0
    with torch.no_grad():
        for features, soh, temp, cycle in test_loader:
            features, soh, temp, cycle = features.to(device), soh.to(device), temp.to(device), cycle.to(device)
            u, _ = model(features, temp, cycle)
            test_mse += torch.nn.MSELoss()(u, soh).item() * features.size(0)
            
    test_mse /= len(test_loader.dataset)
    print(f"Test MSE Loss: {test_mse:.6f}")

if __name__ == '__main__':
    train_model('Dataset_1_NCA_battery', epochs=200, batch_size=256)
