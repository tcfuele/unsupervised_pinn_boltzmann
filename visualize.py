import matplotlib.pyplot as plt
import torch
from utils import *
from bgk_physics import *

def plot_fxv(model, t_fixed, x_grid, v_grid):
    """
    Plot f(x,v) at a fixed time
    """
    t = torch.full((x_grid.size(0),1), t_fixed, device=x_grid.device)
    txv = make_txv_stack(t, x_grid, v_grid)

    f = model(txv).detach().cpu().numpy()
    N_x = x_grid.size(0)
    N_v = v_grid.size(0)

    f_grid = f.reshape(N_x, N_v)  # x along rows, v along columns

    v_grid = v_grid.detach().cpu().numpy()
    x_grid = x_grid.detach().cpu().numpy()

    plt.figure(figsize=(6,5))
    plt.imshow(f_grid, extent=[v_grid[0], v_grid[-1], x_grid[0], x_grid[-1]],
               origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='f(x,v)')
    plt.xlabel('v')
    plt.ylabel('x')
    plt.title(f'f(x,v) at t={t_fixed:.2f}')
    plt.show()

def plot_moments(model, t_fixed, x_grid, v_grid):
    t = torch.full((x_grid.size(0),1), t_fixed, device=x_grid.device)
    txv = make_txv_stack(t, x_grid, v_grid)

    f = model(txv).detach()
    N_x = x_grid.size(0)
    N_v = v_grid.size(0)
    f_grid = f.view(N_x, N_v)

    # Compute moments
    rho = density(f_grid, v_grid)
    u = compute_u(f_grid, v_grid)
    T = temperature(f_grid, v_grid)

    v_grid = v_grid.detach().cpu().numpy()
    x_grid = x_grid.detach().cpu().numpy()
    rho = rho.detach().cpu().numpy()
    u = u.detach().cpu().numpy()
    T = T.detach().cpu().numpy()

    # Plot
    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.plot(x_grid, rho)
    plt.xlabel('x'); plt.ylabel('density ρ'); plt.title('Density')

    plt.subplot(1,3,2)
    plt.plot(x_grid, u)
    plt.xlabel('x'); plt.ylabel('velocity u'); plt.title('Mean velocity')

    plt.subplot(1,3,3)
    plt.plot(x_grid, T)
    plt.xlabel('x'); plt.ylabel('temperature T'); plt.title('Temperature')

    plt.tight_layout()
    plt.show()

from matplotlib.animation import FuncAnimation

def animate_density(model, x_grid, v_grid, t_list):
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(x_grid[0].item(), x_grid[-1].item())
    ax.set_ylim(0, 2.0)  # adjust depending on expected rho

    def update(frame):
        t = t_list[frame]
        t_tensor = torch.full((x_grid.size(0),1), t, device=x_grid.device)
        txv = make_txv_stack(t_tensor, x_grid, v_grid)
        f = model(txv).detach()
        N_x = x_grid.size(0)
        N_v = v_grid.size(0)
        f_grid = f.view(N_x, N_v)
        rho = density(f_grid, v_grid)
        line.set_data(x_grid.cpu(), rho.cpu())
        ax.set_title(f"Density ρ at t={t:.2f}")
        return line,

    ani = FuncAnimation(fig, update, frames=len(t_list), blit=True)
    plt.show()
