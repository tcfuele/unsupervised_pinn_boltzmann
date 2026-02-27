import torch
import models
from math import pi as PI
from bgk_physics import maxwellian
from pinn import bgk_residual
from utils import initialize_physics_data, make_txv_stack, numpy_data_grid
from visualize import plot_fxv, plot_moments, animate_density

torch.set_default_dtype(torch.float64)
#Learning Params
DEVICE = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
EPOCHS = 10000
LR = 1e-5
DTYPE = torch.float64

#Sampling params
N_TX = 100      #Number of points in t and x dimension
N_V = 48       #Number of points in v

X_MIN = -1.0
X_MAX =  1.0
T_MIN =  0.0
T_MAX =  1.0
T0    = 1.0
V_MAX = 6.5 # 5 * (T0 ** 0.5)

TAU = 0.5       #Relaxation time


t, x, v, domain = initialize_physics_data(N_TX, N_V, T_MIN, T_MAX, X_MIN, X_MAX, V_MAX, DEVICE, DTYPE)
txv = make_txv_stack(t, x, v)
print("Learning on domain: ", domain)

model = models.Mlp()
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

x_ic = torch.linspace(X_MIN, X_MAX, N_TX, device=DEVICE).unsqueeze(1)
t_ic = torch.zeros_like(x_ic)
txv_ic = make_txv_stack(t_ic, x_ic, v)

def pde_loss(model, txv, v, tau):

    R = bgk_residual(model, txv, v, tau)

    loss_pde = torch.mean(torch.pow(R, 2))

    return loss_pde

def initial_distribution(txv, rho0=1.0, u0=0.0, T0=1.0, eps=0.05):

    x = txv[:, 1]
    v = txv[:, 2]       #Better sample new? In theory it does not make any difference!

    rho = rho0 * (1.0 + eps * torch.sin(PI * (x - X_MIN) / (X_MAX - X_MIN)))

    prefactor = rho / torch.sqrt(torch.tensor(2.0 * PI * T0, dtype=DTYPE, device=DEVICE))
    exponent = - (v - u0)**2 / (2.0 * T0)

    return prefactor * torch.exp(exponent)

def ic_loss(model):

    f_pred = model(txv_ic)
    f_true = initial_distribution(txv_ic)

    return torch.mean(torch.pow((f_pred - f_true), 2))


def boundary_loss(model, t, v, x_min=-2.0, x_max=2.0):

    x_left = torch.full_like(t, x_min)
    x_right = torch.full_like(t, x_max)

    txv_left = make_txv_stack(t, x_left, v)
    txv_right = make_txv_stack(t, x_right, v)

    f_left = model(txv_left)
    f_right = model(txv_right)

    return torch.mean(torch.pow((f_left - f_right), 2))


for epoch in range(EPOCHS):

    optimizer.zero_grad()

    t = torch.rand(N_TX, 1, device=DEVICE) * (T_MAX - T_MIN) + T_MIN
    x = torch.rand(N_TX, 1, device=DEVICE) * (X_MAX - X_MIN) + X_MIN

    txv = make_txv_stack(t, x, v)

    loss = (
            0.05 * pde_loss(model, txv, v, TAU)              #MAGIC NUMBER!
            + 20 * ic_loss(model) 
            + boundary_loss(model, t, v, X_MIN, X_MAX)
    )

    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6e}")

print("Training complete.")

torch.save(model.state_dict(), f'model_weights{EPOCHS}.pth')

#model.load_state_dict(torch.load("./model_weights10000.pth", weights_only=True))
print("Visualizing!")

#t, x, v, domain = numpy_data_grid(N_TX, N_V, T_MIN, T_MAX, X_MIN, X_MIN, V_MAX)
_, x, v, domain = initialize_physics_data(N_TX, N_V, T_MIN, T_MAX, X_MIN, X_MAX, V_MAX, DEVICE, DTYPE)

x = torch.linspace(X_MIN, X_MAX, N_TX, device=DEVICE).unsqueeze(1)
v = torch.linspace(-V_MAX, V_MAX, N_V, device=DEVICE)
t = 0.0
plot_fxv(model, t, x, v)
plot_moments(model, t, x, v)
animate_density(model, x, v, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
