# unsupervised_pinn_boltzmann

Can a physics-informed neural network approximate the solution of the 1D Boltzmann–BGK equation without supervised data while preserving macroscopic conservation laws?

Plan:
- 1. Build MLP model ✓
    - Forward pass ✓
    - Gradient verification ✓
- 2. Maxwellian computation ✓
    - Moment computation ✓
    - Validation of equilibrium ✓
- 3. PDE residual with autograd ✓
    - Trivial testcase (equilibrium solution?)
- 4. Training loop, first run ✓
    - Initial condition loss ✓
- 5. Add periodic boundary conditions
- 6. Improve Training parameters
- 7. Add visualization and conservation monitoring
- 8. Cleanup! Code, Readme (This project was never in a messy state :D)
