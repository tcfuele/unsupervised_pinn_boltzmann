import torch
import numpy as np

def initialize_physics_data(
    N_tx=1024,
    N_v=48,
    t_min=0.0,
    t_max=5.0,
    x_min=-4.0,
    x_max=4.0,
    v_max=8.2,
    device='cuda',
    dtype=torch.float64,
):
    """
    Initialize physics data for 1D Boltzmann-BGK PINN.

    Returns:
        dict containing:
            t_tx       : (N_tx, 1) space-time time samples
            x_tx       : (N_tx, 1) space-time spatial samples
            v_grid     : (N_v,) velocity grid
            domain     : dictionary with domain metadata
    """


    #Sample space-time points as
    #uniform random collocation points
    t_tx = torch.rand(N_tx, 1, dtype=dtype, device=device).requires_grad_(True)
    t_tx = t_min + (t_max - t_min) * t_tx

    x_tx = torch.rand(N_tx, 1, dtype=dtype, device=device).requires_grad_(True)
    x_tx = x_min + (x_max - x_min) * x_tx

    v_grid = torch.linspace(
        -v_max, v_max, N_v, dtype=dtype, device=device
    ).requires_grad_(True)

    #Domain metadata
    domain = {
        "t_min": t_min,
        "t_max": t_max,
        "x_min": x_min,
        "x_max": x_max,
        "v_max": v_max,
        "N_tx": N_tx,
        "N_v": N_v,
    }

    return t_tx, x_tx, v_grid, domain


def make_txv_stack(t, x, v):
    # Expand physics batch
    N_v = v.size(dim=0)
    N_tx = t.size(dim=0)

    #Repeat is not the most elegant, but okay for a small
    #project like this, in production would change to expand.
    #but did not know if we would changes values (expand returns view!)
    T = t.repeat(1, N_v)
    X = x.repeat(1, N_v)
    V = v.unsqueeze(0).repeat(N_tx, 1)

    return torch.stack([T, X, V], dim=-1).reshape(-1, 3).requires_grad_(True)
