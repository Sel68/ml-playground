import torch
import torch.nn as nn

def gradients(u, x, order=1):
    for _ in range(order):
        u = torch.autograd.grad(u, x, 
                                grad_outputs=torch.ones_like(u), 
                                create_graph=True, retain_graph=True)[0]
    return u

def trainPinn(model, X_u, u, X_f, nu, epochs=5000, lamb=1.0):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    X_u.requires_grad_()

    for _ in range(epochs):
        optimizer.zero_grad()

        # Supervised loss (boundary + initial)
        u_pred = model(X_u)
        mse_u = nn.MSELoss()(u_pred, u)

        # PDE residual loss
        t_f = X_f[:,0:1].clone().detach().requires_grad_(True)
        x_f = X_f[:,1:2].clone().detach().requires_grad_(True)

        u_f = model(torch.cat([t_f, x_f], dim=1))
        u_t = gradients(u_f, t_f)
        u_x = gradients(u_f, x_f)
        u_xx = gradients(u_f, x_f, order=2)

        f = u_t + u_f * u_x - nu * u_xx
        mse_f = torch.mean(f**2)

        loss = mse_u + lamb * mse_f #lamb: balance fitting data and obeying PDE
        loss.backward()
        optimizer.step()
    return loss
