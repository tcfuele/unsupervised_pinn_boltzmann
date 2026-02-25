import torch
from torch.functional import _return_inverse
import torch.nn as nn
import models
from bgk_physics import density, momentum_density, energy_density, compute_u, temperature
from math import pi as PI
from math import sqrt

def compute_maxwellian_fom_model(model, tx_grid, v_grid):

    N = tx_grid.shape[0]
    Nv = v_grid.shape[0]

    #Need to expand t over velocity grid
    tx_expanded = tx_grid.unsqueeze(1).repeat(1, Nv, 1) # (Nt, Nv, 2)
    v_expanded = v_grid.unsqueeze(0).repeat(N, 1).unsqueeze(-1) # (Nt, Nv, 1)

    txv = torch.cat([tx_expanded, v_expanded], dim=2) # ( N, Nv, 3 )
    txv_flat = txv.reshape(-1, 3)

    f_vals = model(txv_flat).reshape(N, Nv)

    rho = density(f_vals, v_grid).unsqueeze(1)

    u = compute_u(f_vals, v_grid).unsqueeze(1)

    T = temperature(f_vals, v_grid).unsqueeze(1)
    v_grid = v_grid.unsqueeze(0)

    return rho / torch.sqrt(2 * PI * T) * torch.exp(torch.pow((v_grid - u), 2) / 2 * T)



def bgk_residual(model, txv, tau, v_grid):

    txv.requires_grad_(True)

    f = model(txv)

    grads = torch.autograd.grad(
        f,
        txv,
        grad_outputs=torch.ones_like(f),
        create_graph=True
    )[0]

    df_dt = grads[:, 0]
    df_dx = grads[:, 1]

    v = txv[:, 2]
    tx = txv[:, :2]

    unique_tx, inverse_indices = torch.unique(tx, dim=0, return_inverse = True)

    M = torch.zeros_like(f)

    for i, tx_val in enumerate(unique_tx):

        mask = (inverse_indices == i)

        f_subset = f[mask]
        v_subset = txv[mask, 2]

        # Change this later, just for convenience
        rho = torch.trapz(f_subset, v_subset)
        rho_u = torch.trapz(v_subset * f_subset, v_subset)
        u = rho_u / rho

        E = torch.trapz(0.5 * v_subset**2 * f_subset, v_subset)
        T = 2 * (E - 0.5 * rho * u**2) / rho

        # Maxwellian evaluated at v_subset
        prefactor = rho / torch.sqrt(2 * PI * T)
        exponent = - (v_subset - u)**2 / (2 * T)
        M_subset = prefactor * torch.exp(exponent)

        M[mask] = M_subset

    residual = df_dt + v * df_dx - (1.0 / tau) * (M - f)

    return residual


class ConstantMaxwellian(nn.Module):
    """
    Returns a fixed Maxwellian distribution independent of (t,x).
    This represents a trivial equilibrium solution.
    """

    def __init__(self, rho=1.0, u=0.0, T=1.0):
        super().__init__()
        self.rho = rho
        self.u = u
        self.T = T

    def forward(self, txv):
        """
        txv: tensor of shape (N, 3) containing (t, x, v)
        """
        v = txv[:, 2]
        prefactor = self.rho / sqrt(2 * PI * self.T)
        exponent = - (v - self.u)**2 / (2 * self.T)
        return prefactor * torch.exp(exponent)

if __name__ == "__main__":

    tau = 1.0

    model = ConstantMaxwellian(rho=1.0, u=0.0, T=1.0)

    # Build test grid
    N_tx = 5
    Nv = 32

    t = torch.zeros(N_tx)
    x = torch.linspace(-1, 1, N_tx)
    v_grid = torch.linspace(-10, 10, Nv)

    tx = torch.stack([t, x], dim=1)

    # Expand into full (t,x,v) grid
    tx_expanded = tx.unsqueeze(1).repeat(1, Nv, 1)
    v_expanded = v_grid.unsqueeze(0).repeat(N_tx, 1).unsqueeze(-1)

    txv = torch.cat([tx_expanded, v_expanded], dim=2).reshape(-1, 3)

    R = bgk_residual(model, txv, tau, v_grid)

    print("Residual L2 norm:", torch.norm(R).item())
