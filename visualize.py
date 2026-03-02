import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from models import Mlp
from matplotlib.animation import FuncAnimation
from utils import *
from bgk_physics import *

torch.set_default_dtype(torch.float64)

def plot_fxv(f, t_fixed, x_grid, v_grid, f_path):
    """
    Plot f(x,v) at a fixed time
    Takes numpy arrays of quantities
    """

    N_x = x_grid.shape[0]
    N_v = v_grid.shape[0]

    f_grid = f.reshape(N_x, N_v)  # x along rows, v along columns
    f_max = f.max()

    norm = mpl.colors.Normalize(vmin=0.0, vmax=f_max, clip=False)

    plt.figure(figsize=(6,5))
    plt.imshow(f_grid, extent=[v_grid[0], v_grid[-1], x_grid[0], x_grid[-1]],
               origin='lower', aspect='auto', cmap='viridis', norm=norm)

    plt.colorbar(label='f(x,v)')
    plt.xlabel('v')
    plt.ylabel('x')
    plt.title(f'f(x,v) at t={t_fixed:.2f}')
    plt.savefig(f_path)
    plt.show()

def plot_moments(rho, u, T, t_fixed, x_grid, moments_path):
    """
    Plots moments at fixed time
    Takes numpy arrays as quantities
    """
    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.plot(x_grid, rho)
    plt.xlabel('x'); plt.ylabel('density ρ'); plt.title(f"Density at t={t_fixed}")

    plt.subplot(1,3,2)
    plt.plot(x_grid, u)
    plt.xlabel('x'); plt.ylabel('velocity u'); plt.title(f"Mean velocity at t={t_fixed}")

    plt.subplot(1,3,3)
    plt.plot(x_grid, T)
    plt.xlabel('x'); plt.ylabel('temperature T'); plt.title(f"Temperature at t={t_fixed}")

    plt.tight_layout()
    plt.savefig(moments_path)
    plt.show()

"""
def animate_density(model, x_grid, v_grid, t_list):
    model.eval()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(x_grid[0].item(), x_grid[-1].item())
    ax.set_ylim(0, 2.0)  # adjust depending on expected rho

    def update(frame):
        t = t_list[frame]
        t_tensor = torch.full(
            (x_grid.size(0), 1),
            t,
            device=x_grid.device
        )
        txv = make_txv_stack(t_tensor, x_grid, v_grid)
        with torch.no_grad():
            f = model(txv)
        N_x = x_grid.size(0)
        N_v = v_grid.size(0)
        f_grid = f.view(N_x, N_v)
        rho = density(f_grid, v_grid)
        line.set_data(x_grid.detach().cpu().numpy(), rho.detach().cpu().numpy())
        ax.set_title(f"Density ρ at t={t:.2f}")
        return line

    ani = FuncAnimation(fig, update, frames=len(t_list), blit=False)
    plt.show()

"""
def animate_density(model, x_grid, v_grid, t_list, ani_path):
    N_x = x_grid.size(0)
    N_v = v_grid.size(0)

    # Precompute first frame
    t0 = t_list[0]
    t_tensor = torch.full((N_x,1), t0, device=x_grid.device)
    txv = make_txv_stack(t_tensor, x_grid, v_grid)
    with torch.no_grad():
        f = model(txv)
    f_grid = torch.stack(f.split(N_v))
    rho0 = density(f_grid, v_grid)

    fig, ax = plt.subplots()
    line, = ax.plot(x_grid.detach().cpu().numpy(),
                    rho0.detach().cpu().numpy(), lw=2)
    ax.set_xlim(x_grid[0].item(), x_grid[-1].item())
    ax.set_ylim(0, 1.1 * rho0.max().item())  # scale to first frame

    def update(frame):
        t = t_list[frame]
        t_tensor = torch.full((N_x,1), t, device=x_grid.device)
        txv = make_txv_stack(t_tensor, x_grid, v_grid)
        with torch.no_grad():
            f = model(txv)
        f_grid = torch.stack(f.split(N_v))
        rho = density(f_grid, v_grid)
        line.set_data(x_grid.detach().cpu().numpy(),
                      rho.detach().cpu().numpy())
        ax.set_title(f"Density ρ at t={t:.2f}")
        return line,

    ani = FuncAnimation(fig, update, interval=500, frames=len(t_list), blit=False)
    ani.save(ani_path)
    plt.show()


#Execute as main for visualization
if __name__ == "__main__":
    import torch
    import models
    from pathlib import Path

    DIR_PATH = Path("./plots")
    WEIGHTS = 12000
    DEVICE = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    DTYPE = torch.float64

    #Sampling parameters
    #Grid parameters
    N_TX = 128      #Number of points in t and x dimension
    N_V = 56        #Number of points in v

    X_MIN = -1.0    #spatial grid
    X_MAX =  1.0
    T_MIN =  0.0    #temporal grid
    T_MAX =  1.0
    T0    = 1.0     #temperature for 
    V_MAX = 6.5 # 5 * (T0 ** 0.5)

    t_plot = 0.1
    t_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    DIR_PATH.mkdir(exist_ok=True)

    f_path = DIR_PATH.joinpath(f"f_{WEIGHTS}.jpg")
    moments_path = DIR_PATH.joinpath(f"moments_{WEIGHTS}.jpg")
    ani_path = DIR_PATH.joinpath(f"ani_{WEIGHTS}.gif")

    model = Mlp().to(DEVICE)
    model.load_state_dict(torch.load(f"./model_weights{WEIGHTS}.pth", weights_only=True))
    model.eval()

    t_ten = torch.linspace(-T_MIN, T_MAX, N_TX, device=DEVICE).unsqueeze(1)
    x_ten = torch.linspace(X_MIN, X_MAX, N_TX, device=DEVICE).unsqueeze(1)
    v_ten = torch.linspace(-V_MAX, V_MAX, N_V, device=DEVICE)
    txv = make_txv_stack(t_ten, x_ten, v_ten)

    f = model(txv).reshape(N_TX, N_V)
    rho = density(f, v_ten.unsqueeze(0)).detach().cpu().numpy()
    u = density(f, v_ten.unsqueeze(0)).detach().cpu().numpy()
    temperature = temperature(f, v_ten.unsqueeze(0)).detach().cpu().numpy()


    t = np.linspace(T_MIN, T_MAX, N_TX)
    x = np.linspace(X_MIN, X_MAX, N_TX)
    v = np.linspace(-V_MAX, V_MAX, N_V)

    f_grid = model(txv).detach().cpu().numpy()

    plot_fxv(f_grid, t_plot, x, v, f_path)
    plot_moments(rho, u, temperature, t_plot, x, moments_path)
    animate_density(model, x_ten, v_ten, t_list, ani_path)
