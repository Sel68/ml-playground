import torch
import numpy as np
from scipy.stats import qmc

def get_training_data(N0 = 100, Nb = 100, Nf = 10000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Initial Conds
    x0 = np.random.uniform(-1, 1, (N0, 1))
    t0 = np.zeros_like(x0)
    u0 = -np.sin(np.pi * x0)

    #Boundary Conds
    tb = np.random.uniform(0, 1, (Nb, 1))
    xb1 = -np.ones_like(tb)
    xb2 = np.ones_like(tb)
    ub1 = np.zeros_like(tb)
    ub2 = np.zeros_like(tb)

    sampler = qmc.LatinHypercube(d=2)
    X_f = sampler.random(Nf)
    X_f = np.array([X_f[:,1]*2 - 1, X_f[:,0]]).T

    X_u = np.vstack([np.hstack([t0, x0]),
                     np.hstack([tb, xb1]),
                     np.hstack([tb, xb2])])
    u = np.vstack([u0, ub1, ub2])

    return torch.tensor(X_u, dtype=torch.float32).to(device), \
           torch.tensor(u, dtype=torch.float32).to(device), \
           torch.tensor(X_f, dtype=torch.float32).to(device)
