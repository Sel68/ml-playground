import numpy as np

import numpy as np

def burgers_data(Nx=256, final_time=1.0, nu=0.01, safety_conv=0.2, safety_diff=0.4):
    """
    Stable explicit FD solver for 1D viscous Burgers with periodic BCs.
    Chooses dt = min(dt_conv, dt_diff) with safety factors.
    """
    x = np.linspace(0.0, 1.0, Nx)
    dx = x[1] - x[0]

    u_init = -np.sin(np.pi * x)
    max_u = np.max(np.abs(u_init))

    # convection-based dt (avoid division by zero)
    if max_u < 1e-12:
        dt_conv = np.inf
    else:
        dt_conv = safety_conv * dx / max_u

    # diffusion-based dt (FTCS stability: dt <= dx^2 / (2*nu))
    if nu <= 0:
        dt_diff = np.inf
    else:
        dt_diff = safety_diff * dx * dx / (2.0 * nu)

    dt = min(dt_conv, dt_diff)
    if not np.isfinite(dt) or dt <= 0:
        raise RuntimeError("Computed dt is invalid. Check Nx, nu, or safety factors.")

    Nt = max(2, int(np.ceil(final_time / dt)) + 1)
    t = np.linspace(0.0, final_time, Nt)
    dt = t[1] - t[0]  # actual dt used

    u = u_init.copy()
    U = np.zeros((Nt, Nx), dtype=float)
    U[0, :] = u.copy()

    for n in range(1, Nt):
        un = u.copy()
        u_x = (np.roll(un, -1) - np.roll(un, 1)) / (2.0 * dx)
        u_xx = (np.roll(un, -1) - 2.0 * un + np.roll(un, 1)) / (dx * dx)

        u = un - dt * un * u_x + dt * nu * u_xx

        if np.any(np.isnan(u)) or np.any(np.isinf(u)):
            raise RuntimeError(f"Simulation blew up at step {n}. Try smaller safety_conv/safety_diff or increase Nx.")
        U[n, :] = u.copy()

    X, T = np.meshgrid(x, t)
    return X.flatten()[:, None], T.flatten()[:, None], U.flatten()[:, None]



def sparse_sample(X, T, U, N_points=500, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    idx = rng.choice(len(X), size=N_points, replace=False)
    return X[idx], T[idx], U[idx]
