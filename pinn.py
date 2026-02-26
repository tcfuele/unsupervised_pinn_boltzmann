import torch
import torch.nn as nn
from math import pi as PI, sqrt
from bgk_physics import density, compute_u, temperature, maxwellian


def bgk_residual(model, txv, tau=1.0):
    """
    Computes the BGK PDE residual at given (t,x,v) points.
    txv: shape (N,3), columns are (t, x, v)

    Returns R: shape (N,)
    """
    print("txv shape",txv.shape)
    #txv = txv.clone().detach().requires_grad_(True)
    f = model(txv)
    print("f dim", f.dim())

    # Gradients
    grads = torch.autograd.grad(
        f, txv,
        grad_outputs=torch.ones_like(f),
        create_graph=True
    )[0]

    N_total = txv.size(0)
    df_dt = grads[:, 0]  # d f / dt
    df_dx = grads[:, 1]  # d f / dx
    v = txv[:, 2]        # velocity
    v_grid = torch.unique(txv[:, 2].detach())
    N_v = v_grid.size(0)
    N_tx = N_total // N_v

    f_grid = f.view(N_tx, N_v)
    # Compute macroscopic quantities for each point
    # Here, for simplicity, we treat each point independently
    rho = density(f_grid, v_grid)       # scalar per point
    u = compute_u(f_grid, v_grid).repeat_interleave(N_v)
    rho = rho.repeat_interleave(N_v)
    T = temperature(f_grid, v_grid).repeat_interleave(N_v)
    M = maxwellian(rho, u, T, v)

    # BGK PDE residual
    R = df_dt + v * df_dx - (1.0 / tau) * (M - f)
    return R


class ConstantMaxwellian(nn.Module):
    """
    Returns a fixed Maxwellian distribution independent of (t,x,v)
    """

    def __init__(self, rho=1.0, u=0.0, T=1.0):
        super().__init__()
        self.rho = rho
        self.u = u
        self.T = T

    def forward(self, txv):
        v = torch.unique(txv[:, 2])
        prefactor = self.rho / sqrt(2 * PI * self.T)
        exponent = - (v - self.u)**2 / (2 * self.T)
        return prefactor * torch.exp(exponent)


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

    R = bgk_residual(model, txv, tau)
    print("Residual L2 norm:", torch.norm(R).item())
