# unsupervised_pinn_boltzmann

This project was created to answer the question: "Can a physics-informed neural network approximate the solution of the 1D Boltzmann–BGK equation without supervised data while preserving macroscopic conservation laws?"


It is indeed possible to solve this and there is very interesting insight to be gained from that. Currently I am working on a physics notebook to provide further information on that, as well as tweaking the parameters. Otherwise the model is fully functional and you can run it on your own machine (see Usage).

The model will train an unsupervised PINN with loss terms enforcing right behaviour of: PDE, boundary condition, initial condition and moment conservation.
The initial condition looks like a flat density distribution with a small cosine perturbation:

## Results

![](https://github.com/tcfuele/unsupervised_pinn_boltzmann/blob/main/figures/moments_12000.jpg)


One can see an offset here since the parameters for the boundary loss are not perfectly fixed yet. The PDE is translational invariant, the multi layer perceptron is not! That's where more tweaking is needed.
The learned distribution at initial time can be visualized in the phase space like this:

![](https://github.com/tcfuele/unsupervised_pinn_boltzmann/blob/main/figures/f_12000.jpg)


Also I provide a small gif generator to look at the time evolution of the density function of the system:

![](https://github.com/tcfuele/unsupervised_pinn_boltzmann/blob/main/figures/ani_12000.gif)



## Usage

The model is run by executing `training.py` as main:

```
python3 training.py
```

All simulation parameters can be found at the top of `training.py` and tweaked from there. I provide the parameters that the plots in this README were created with.

Plots can be created by running `visualize.py` as main. In doing so you might have to tweak plotting parameters or even the output directory in the at the end of the file.
All other modules complete their tests when run as main.

## TODO

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
- 5. Add periodic boundary conditions ✓
- 6. Add visualization and conservation monitoring ✓
- 7. Reduce memory load ✓
- 8. Improve Training parameters
- 9. Cleanup! Code, Readme (This project was never in a messy state :D)
- 10. Physics notebook for further explanation
