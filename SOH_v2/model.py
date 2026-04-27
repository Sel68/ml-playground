import torch
import torch.nn as nn

class SolutionNetwork(nn.Module):
    def __init__(self, n_features):
        super(SolutionNetwork, self).__init__()
        # Input: t_arr (1) + features (n_features)
        self.net = nn.Sequential(
            nn.Linear(1 + n_features, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
    def forward(self, t_arr, x):
        inputs = torch.cat([t_arr, x], dim=1)
        return self.net(inputs)

class DynamicsNetwork(nn.Module):
    def __init__(self, n_features):
        super(DynamicsNetwork, self).__init__()
        # Input: t_arr (1) + features (n_features) + u (1)
        self.net = nn.Sequential(
            nn.Linear(1 + n_features + 1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
    def forward(self, t_arr, x, u):
        inputs = torch.cat([t_arr, x, u], dim=1)
        return self.net(inputs)

class BatteryPINN(nn.Module):
    def __init__(self, n_features=19):
        super(BatteryPINN, self).__init__()
        self.F_net = SolutionNetwork(n_features)
        self.G_net = DynamicsNetwork(n_features)
        
        # Trainable Ea initialized at 52.0 kJ/mol
        self.Ea = nn.Parameter(torch.tensor([52.0], dtype=torch.float32))
        self.R = 8.314e-3  # kJ/(mol*K)
        self.T_ref = 298.15  # K
        
    def compute_t_arr(self, temp_c, cycle):
        # temp_c is in Celsius
        T_i = temp_c + 273.15
        
        # Arrhenius factor
        # gamma(T) = exp[ (Ea / R) * (1/T_ref - 1/T) ]
        gamma = torch.exp((self.Ea / self.R) * (1.0 / self.T_ref - 1.0 / T_i))
        
        # t_arr = sum(gamma * delta_t). Since temperature is constant per cell and cycle delta=1
        t_arr = cycle * gamma
        return t_arr
        
    def forward(self, x, temp_c, cycle):
        # We need t_arr to require gradients to compute L_PDE
        # But cycle doesn't have gradients. We compute t_arr and then make it require_grad=True
        # Actually it's better if we just derive wrt t_arr directly
        
        T_i = temp_c + 273.15
        gamma = torch.exp((self.Ea / self.R) * (1.0 / self.T_ref - 1.0 / T_i))
        t_arr = cycle * gamma
        
        # To compute gradients dF/dt_arr, t_arr must require grad
        # In training loop we will handle this.
        
        u = self.F_net(t_arr, x)
        return u, t_arr

def compute_pinn_losses(model, x, temp_c, cycle, true_u, alpha=0.1, beta=0.1):
    # Ensure t_arr requires gradient
    t_arr = model.compute_t_arr(temp_c, cycle)
    # If t_arr is not a leaf but we need gradient w.r.t it:
    if not t_arr.requires_grad:
        t_arr.requires_grad_(True)
    else:
        t_arr.retain_grad()
    
    u = model.F_net(t_arr, x)
    
    # 1. Data Loss
    mse_loss = nn.MSELoss()(u, true_u)
    
    # 2. PDE Loss
    # L_PDE: Enforces |dF/dt_arr - G| = 0
    # Compute du/dt_arr
    u_t = torch.autograd.grad(
        u, t_arr, 
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]
    
    G_pred = model.G_net(t_arr, x, u)
    pde_loss = torch.mean((u_t - G_pred)**2)
    
    # 3. Monotonicity Loss
    # Ensure SOH is non-increasing (u_{k+1} <= u_k). This translates to u_t <= 0 (since t_arr increases with cycles)
    # L_mono = mean( ReLU(u_t) )
    mono_loss = torch.mean(torch.relu(u_t))
    
    loss = mse_loss + alpha * pde_loss + beta * mono_loss
    
    return loss, mse_loss, pde_loss, mono_loss
