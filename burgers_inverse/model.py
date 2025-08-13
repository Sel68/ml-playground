import torch.nn as nn
import torch

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
    u = model(x, t)

    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    f = u_t + u * u_x - model.nu * u_xx
    return torch.mean(f**2)