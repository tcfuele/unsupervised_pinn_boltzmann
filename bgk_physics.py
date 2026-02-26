import torch
from numpy import pi as PI

torch.set_default_dtype(torch.float32)
#It is important, that v range is symmetric AND wide enough (v = u plusminus 5 sqrt(T)

def density(f, v):

    return torch.trapezoid(f, v, dim=1)


def momentum_density(f, v):

    return torch.trapezoid(f * v, v, dim=1)


def energy_density(f, v):

    return torch.trapezoid(0.5 * f * torch.pow(v , 2), v, dim=1)


def compute_u(f, v):

    rho_safe = density(f, v) + 1e-12 #takes care for the case when rho is 0 :)

    return momentum_density(f, v) / rho_safe


def temperature(f, v):

    rho_safe = density(f, v) + 1e-12 #takes care for the case when rho is 0 :)
    u = compute_u(f, v)

    return (2.0 / rho_safe) * (energy_density(f, v) - 0.5 * rho_safe * torch.pow(u, 2))


def maxwellian(rho, u, T, v):



    prefactor = torch.sqrt(2.0 * 3 * T)
    prefactor = rho / torch.sqrt(2.0 * PI * T)
    exponent = - torch.pow((v - u), 2) / (2.0 * T)

    return prefactor * torch.exp(exponent)

def bgk_collision(f, M, tau):

    return (M - f) / tau


def test_maxwellian_reconstruction(v_grid):
    batch_size = 4

    rho = torch.tensor([1.0, 0.8, 1.2, 0.5])
    u = torch.tensor([2.0, 0.5, -0.3, 1.0])
    T = torch.tensor([1.0, 0.7, 1.5, 0.9])

    M = maxwellian(rho, u, T, v_grid)

    rho_rec = density(M, v_grid)
    u_rec = compute_u(M, v_grid)
    T_rec = temperature(M, v_grid)

    M_rec = maxwellian(rho_rec, u_rec, T_rec, v_grid)

    error = torch.norm(M - M_rec) / torch.norm(M)

    print("Reconstruction relative error:", error.item())



if __name__ == "__main__":

    func_tensor = torch.ones((20,num_grid_p))

    rho = density(func_tensor, v_grid)
    u = compute_u(func_tensor, v_grid)
    T = temperature(func_tensor, v_grid)

    test_maxwellian_reconstruction(v_grid)
