import numpy as np

def burgers_data(Nx=256, final_time=1.0, nu=0.01):
    x = np.linspace(0.0, 1.0, Nx)
    dx = x[1] - x[0]

    u = -np.sin(np.pi * x)
    dt = 0.4 * dx*dx / (2*nu) if nu>0 else 0.01
    Nt = max(2, int(np.ceil(final_time/dt))+1)
    t = np.linspace(0.0, final_time, Nt)
    dt = t[1] - t[0]

    U = np.zeros((Nt, Nx))
    U[0] = u.copy()

    for n in range(1, Nt):
        un = u.copy()
        u_x = (np.roll(un,-1)-np.roll(un,1))/(2*dx)
        u_xx = (np.roll(un,-1)-2*un+np.roll(un,1))/(dx*dx)
        u = un - dt*un*u_x + dt*nu*u_xx
        U[n] = u.copy()

    X, T = np.meshgrid(x, t)
    return X.flatten()[:,None], T.flatten()[:,None], U.flatten()[:,None]


def sparse_sample(X, T, U, N_points=500, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    idx = rng.choice(len(X), size=N_points, replace=False)
    return X[idx], T[idx], U[idx]
