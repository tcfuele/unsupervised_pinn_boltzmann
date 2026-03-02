import torch
import torch.nn as nn
from math import pi as PI, sqrt
from bgk_physics import density, compute_u, energy_density, momentum_density, temperature, maxwellian
from utils import make_txv_stack

def bgk_residual(model, t, x, v, tau=1.0):
    """
    Computes residuals of the bgk maxwellian.

    Inputs:
    t : (N_TX, 1)
    x : (N_TX, 1)
    v : (N_V,)

    Outputs:
    R : (N_TX * N_V)
    """

    # Build full Cartesian product (N_tx * N_v, 3)
    txv = make_txv_stack(t, x, v)
    txv = txv.clone().detach().requires_grad_(True)

    N_total = txv.size(0)
    N_v = v.size(0)
    N_tx = N_total // N_v

    # Forward pass
    f = model(txv).squeeze(-1)          # (N_tx * N_v,)
    f_grid = f.view(N_tx, N_v)          # (N_tx, N_v)

    # Compute gradients w.r.t. (t,x,v)
    grads = torch.autograd.grad(
        f,
        txv,
        grad_outputs=torch.ones_like(f),
        create_graph=True
    )[0]

    # Extract time and space derivatives
    df_dt = grads[:, 0].view(N_tx, N_v)
    df_dx = grads[:, 1].view(N_tx, N_v)

    # Expand velocity grid to batch form for maxwellian and residual
    v_batch = v.unsqueeze(0).expand(N_tx, -1)

    rho = density(f_grid, v)
    rho_safe = rho + 1e-12

    momentum = momentum_density(f_grid, v)
    energy = energy_density(f_grid, v)

    u = momentum / rho_safe
    T = (2.0 / rho_safe) * (energy - 0.5 * rho_safe * u**2)

    rho = rho.unsqueeze(1)
    u = u.unsqueeze(1)
    T = T.unsqueeze(1)

    M = maxwellian(rho, u, T, v_batch)

    # --- BGK residual ---
    R = df_dt + v_batch * df_dx - (1.0 / tau) * (M - f_grid)

    return R.reshape(-1)


#Execute as main for testing
if __name__ == "__main__":
    from utils import initialize_physics_data, make_txv_stack

    class TestMaxwellian(nn.Module):
        """
        Analytical test function: rho(x) * Maxwellian(v)
        """

        def __init__(self, rho0=1.0, u0=0.0, T0=1.0, eps=0.01):
            super().__init__()
            self.rho0 = rho0
            self.u0 = u0
            self.T0 = T0
            self.eps = eps

        def forward(self, txv):
            x = txv[:, 1]
            v = txv[:, 2]
            rho = self.rho0 * (1.0 + self.eps * torch.sin(PI * x))
            prefactor = rho / sqrt(2 * PI * self.T0)
            exponent = - (v - self.u0)**2 / (2 * self.T0)
            return prefactor * torch.exp(exponent)

    t, x, v, domain = initialize_physics_data(10, 1024)
    txv = make_txv_stack(t, x, v)

    # --- Test the residual ---
    tau = 1.0
    model = TestMaxwellian(eps=0)

    R = bgk_residual(model, txv, v, tau)
    print("Residual L2 norm:", torch.norm(R).item())
