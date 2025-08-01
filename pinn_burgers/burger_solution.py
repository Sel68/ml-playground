import numpy as np

def exact_burgers_solution(x, t, nu=0.01/np.pi, N=100):
    """
    w Fourier series truncation.
    Parameters:
        nu : viscosity
        N : number of Fourier modes (more -> better)
    Returns:
        u : (n,) array of u(x, t)
    """
    x = np.atleast_1d(x)
        
    phi = np.ones_like(x)
    dphi = np.zeros_like(x)
    
    for n in range(1, N):
        k = n * np.pi
        exp_factor = np.exp(-k**2 * nu * t)
        phi  += 2 * exp_factor * np.cos(k * x)
        dphi -= 2 * k * exp_factor * np.sin(k * x)
    
    u = -2 * nu * dphi / phi
    return u
    
