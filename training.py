import torch
import models
from bgk_physics import *
from bgk_physics import maxwellian
from pinn import bgk_residual
from utils import initialize_physics_data, make_txv_stack

#Hardware specifics
DEVICE = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
DTYPE = torch.float64

#Learning parameters
EPOCHS = 35000
LR = 1e-4

#Dynamical parameters:
TAU = 1.0       #Relaxation time of Boltzmann BGK solution

X_MIN = -1.0    #spatial grid
X_MAX =  1.0

T_MIN =  0.0    #temporal grid
T_MAX =  1.0

T0    = 1.0     #temperature for initial condition

V_MAX = 6.5     #Maximum velocity

#Grid parameters
N_TX = 256      #Number of points in t and x dimension each
N_V = 64        #Number of points in v


def moment_conservation_loss(model, t, x, v):
    txv = make_txv_stack(t, x, v)
    f = model(txv).squeeze(-1)
    N_v = v.size(0)
    N_tx = t.size(0)
    f = f.view(N_tx, N_v)

    rho = density(f, v)
    u = compute_u(f, v)
    T = temperature(f, v)

    M = maxwellian(rho.unsqueeze(1), u.unsqueeze(1), T.unsqueeze(1),
                   v.unsqueeze(0).expand(N_tx, -1))

    mass_error = torch.trapezoid(M - f, v, dim=1)
    momentum_error = torch.trapezoid((M - f) * v, v, dim=1)
    energy_error = torch.trapezoid(0.5*(M - f)*v**2, v, dim=1)

    return (
        torch.mean(mass_error**2)
        + torch.mean(momentum_error**2)
        + torch.mean(energy_error**2)
    )


def pde_loss(model, t, x, v, tau):
    """
    Computes the mean squared loss of the residual from the
    bgk maxwellian. This is the pde loss.
    """
    R = bgk_residual(model, t, x, v, tau)
    loss_pde = torch.mean(torch.pow(R, 2))
    return loss_pde


def initial_distribution(txv, rho0=1.0, u0=0.0, T0=1.0, eps=0.05):
    """
    Initial distribution is a flat density with a small cosine
    disturbance.
    """
    x = txv[:, 1]
    v = txv[:, 2]

    rho = rho0 * (1.0 + eps * torch.cos(torch.pi * x))

    prefactor = rho / torch.sqrt(torch.tensor(2.0 * PI * T0, dtype=DTYPE, device=DEVICE))
    exponent = - (v - u0)**2 / (2.0 * T0)

    return prefactor * torch.exp(exponent)


def ic_loss(model, txv_ic):
    """
    Forces the pinn solution to take the form
    of initial distribution at t=0.0
    """
    f_pred = model(txv_ic).squeeze(-1)
    f_true = initial_distribution(txv_ic)
    return torch.mean(torch.pow((f_pred - f_true), 2))


def boundary_loss(model, t, v, x_min=-2.0, x_max=2.0):
    """
    Fixes pinn solution at the edges of spatial domain.
    Internally forces Pinn to have X_MAX and X_MIN at that
    point.
    """
    x_left = torch.full_like(t, x_min)
    x_right = torch.full_like(t, x_max)

    txv_left = make_txv_stack(t, x_left, v)
    txv_right = make_txv_stack(t, x_right, v)

    f_left = model(txv_left).squeeze(-1)
    f_right = model(txv_right).squeeze(-1)

    return torch.mean(torch.pow((f_left - f_right), 2))


def train_loop(save_wheights=True):
    t, x, v, domain = initialize_physics_data(      #Domain is handy dict for logging (below)
        N_tx=N_TX,
        N_v=N_V,
        t_min=T_MIN,
        t_max=T_MAX,
        x_min=X_MIN,
        x_max=X_MAX,
        v_max=V_MAX,
        device=DEVICE,
        dtype=DTYPE
    )

    x_ic = torch.linspace(X_MIN, X_MAX, N_TX, dtype=DTYPE, device=DEVICE).unsqueeze(1)
    t_ic = torch.zeros_like(x_ic)
    v_ic = v

    txv_ic = make_txv_stack(t_ic, x_ic, v_ic).detach()

    print("Learning on domain: ", domain)

    model = models.Mlp()
    model.to(DEVICE).to(DTYPE)
    model.apply(models.xavier_init)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        t = torch.rand(N_TX, 1, device=DEVICE) * (T_MAX - T_MIN) + T_MIN
        x = torch.rand(N_TX, 1, device=DEVICE) * (X_MAX - X_MIN) + X_MIN

        loss = (
                1 * pde_loss(model, t, x, v, TAU)              #magic numbers, but they work!
              + 50 * ic_loss(model, txv_ic) 
              + 5 * boundary_loss(model, t, v, X_MIN, X_MAX)
              + 1 * moment_conservation_loss(model, t, x, v)
        )

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6e}")

    if(save_wheights):
        torch.save(model.state_dict(), f'model_weights{EPOCHS}.pth')
    print("Training complete.")

#Keeps functions in here modular and importable
if __name__ == "__main__":
    train_loop(save_wheights=True)
