import torch
import models
from math import pi as PI
from bgk_physics import maxwellian
from pinn import bgk_residual
from utils import initialize_physics_data, make_txv_stack

torch.set_default_dtype(torch.float32)
#Learning Params
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
LR = 1e-3
DTYPE = torch.float32
SEED = 42       #Make sure to change to something random, if not testing! :)

#Sampling params
N_TX = 128      #Number of points in t and x dimension
N_V = 96        #Number of points in v

X_MIN = -2.0
X_MAX =  2.0
T_MIN =  0.0
T_MAX =  0.5
V_MAX =  6.0

TAU = 0.1       #Relaxation time


t, x, v, domain = initialize_physics_data(N_TX, N_V, T_MIN, T_MAX, X_MIN, X_MAX, V_MAX, DEVICE, DTYPE, SEED)
txv = make_txv_stack(t, x, v)
print("Learning on domain: ", domain)

model = models.Mlp()
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)


def pde_loss(model, txv, tau):

    R = bgk_residual(model, txv, tau)

    loss_pde = torch.mean(torch.pow(R, 2))

    return loss_pde

def initial_distribution(txv, rho0=1.0, u0=0.0, T0=1.0, eps=0.1):

    x = txv[:, 1]
    v = txv[:, 2]       #Better sample new? In theory it does not make any difference!

    rho = rho0 * (1.0 + eps * torch.sin(PI * x))

    prefactor = rho / torch.sqrt(torch.tensor(2.0 * PI * T0, dtype=DTYPE, device=DEVICE))
    exponent = - (v - u0)**2 / (2.0 * T0)

    return prefactor * torch.exp(exponent)

def ic_loss(model, x, v):
    t0 = torch.zeros_like(x)

    txv0 = make_txv_stack(t0, x, v)

    f_pred = model(txv0)

    f_true = initial_distribution(txv0)

    return torch.mean(torch.pow((f_pred - f_true), 2))

print(txv.shape)

def boundary_loss(model, t, v, x_min=-2.0, x_max=2.0):

    x_left = torch.full_like(t, x_min)
    x_right = torch.full_like(t, x_max)

    txv_left = make_txv_stack(t, x_left, v)
    txv_right = make_txv_stack(t, x_right, v)

    f_left = model(txv_left)
    f_right = model(txv_right)

    return torch.mean(torch.pow((f_left - f_right), 2))


for epoch in range(EPOCHS):
    #y_pred = model(txv)

    optimizer.zero_grad()

    t = torch.rand(N_TX, 1, device=DEVICE)
    x = torch.rand(N_TX, 1, device=DEVICE) * (X_MAX - X_MIN) - X_MAX  # [-2,2]

    txv = make_txv_stack(t, x, v)

    loss = pde_loss(model, txv, TAU) + 10 * ic_loss(model, x, v) + 1.0 * boundary_loss(model, t, v, X_MIN, X_MAX) #MAGIC NUMBER!
    print(loss.item())

    loss.backward()
    optimizer.step()

    #if epoch % 500 == 0:
    print(f"Epoch {epoch}, Loss: {loss.item():.6e}")

print("Training complete.")


