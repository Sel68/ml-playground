import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data import burgers_data, sparse_sample
import os

np.random.seed(0)
torch.manual_seed(0)

Nx = 256
final_time = 1.0
nu_true = 0.03
X, T, U = burgers_data(Nx, final_time, nu_true, safety_conv=0.2, safety_diff=0.4)


if np.any(np.isnan(U)) or np.any(np.isinf(U)):
    raise SystemExit("U contains NaN/Inf; adjust CFL/safety or Nx.")

N_u = 1000
X_u, T_u, U_u = sparse_sample(X, T, U, N_points=N_u, rng=np.random.default_rng(1))

N_f = 2000
idx_f = np.random.choice(len(X), size=N_f, replace=False)
X_f = X[idx_f]; T_f = T[idx_f]

def scale_x(x): return 2.0 * (x - 0.5)
def scale_t(t): return 2.0 * (t / final_time - 0.5)

x_u = torch.tensor(scale_x(X_u), dtype=torch.float32)
t_u = torch.tensor(scale_t(T_u), dtype=torch.float32)
u_u = torch.tensor(U_u, dtype=torch.float32)

x_f = torch.tensor(scale_x(X_f), dtype=torch.float32, requires_grad=True)
t_f = torch.tensor(scale_t(T_f), dtype=torch.float32, requires_grad=True)

# reshape to (N,1)
x_u = x_u.reshape(-1,1); t_u = t_u.reshape(-1,1); u_u = u_u.reshape(-1,1)
x_f = x_f.reshape(-1,1); t_f = t_f.reshape(-1,1)

device = torch.device("cpu")
x_u = x_u.to(device); t_u = t_u.to(device); u_u = u_u.to(device)
x_f = x_f.to(device); t_f = t_f.to(device)


class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.nu = nn.Parameter(torch.tensor([0.01], dtype=torch.float32))
        layer_list = []
        for i in range(len(layers)-1):
            layer_list.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers)-2:
                layer_list.append(nn.Tanh())
        self.net = nn.Sequential(*layer_list)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)

def pde_loss(model, x, t):
    # x,t: require_grad True
    u = model(x, t)

    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    f = u_t + u * u_x - model.nu * u_xx
    return torch.mean(f**2)

layers = [2, 64, 64, 64, 1]
model = PINN(layers).to(device)

# optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# training hyperparams
epochs = 5000
print_every = 500
lambda_pde = 2.0
lambda_data = 1.0

for epoch in range(epochs+1):
    model.train()
    optimizer.zero_grad()

    # data loss
    pred_u = model(x_u, t_u)
    loss_u = torch.mean((pred_u - u_u)**2)

    x_f.requires_grad_(True); t_f.requires_grad_(True)
    loss_f = pde_loss(model, x_f, t_f)

    loss = lambda_data * loss_u + lambda_pde * loss_f
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"NaN/Inf loss at epoch {epoch}. Aborting.")
        break

    loss.backward()
    # gradient clipping to avoid explosion
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if epoch % print_every == 0:
        nu_val = float(model.nu.detach().cpu().numpy())
        print(f"Epoch {epoch}, Loss: {loss.item():.6e}, loss_u: {loss_u.item():.6e}, loss_f: {loss_f.item():.6e}, nu: {nu_val:.6e}")

os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/pinn_burgers.pth")
print("Training finished, model saved to checkpoints/pinn_burgers.pth")
