# Variations on the PFHub BM 1 Initial Conditions

Source code (Python and Jupyter) for PFHub BM 1a (periodic spinodal
decomposition) using Steppyngstounes and a semi-implicit spectral solver,
with variations on the initial condition.

* **orig** is the "original" IC, not periodic at all
* **peri** is a purely periodic IC, with coefficients
  as close to the original as could be managed
* **zany** is close to periodic, but not, to ward off
  simple periodic pattern formation

The coefficients are explored in `initial_conditions.py`.

## Discretization

> Thanks to Nana Ofori-Opoku (McMaster University) for fruitful
> exposition of the nuances of the spectral method for Cahn-Hilliard.

Broadly, the Cahn-Hilliard equation of motion is

$$
\frac{\partial c}{\partial t} =
  M\nabla^2\left(\frac{\partial f}{\partial c} - \kappa \nabla^2 c\right)
$$

Using the Fourier transform from real to reciprocal space means convolutions
(e.g., $\nabla c$ and scalar multiplication) become multiplications in
reciprocal space, while exponents in real space (i.e., $c^{n\neq 1}$) become
convolutions in reciprocal space. The former simplifies life; the latter does
not. In practice, convolutions are transformed, and non-linear terms are solved
in real space and then transformed. Specifically,

$$ \mathfrak{F}\left[\nabla c\right] = i\vec{k}\hat{c} $$

$$ \mathfrak{F}\left[\nabla^2 c\right] = -\vec{k}^2 \hat{c}$$

$$ \mathfrak{F}\left[\mathrm{const}\right] = \mathrm{const} $$

Transforming the equation of motion, we have

$$
\frac{\partial \hat{c}}{\partial t} = - M\vec{k}^2\left(
\mathfrak{F}\left[\frac{\partial f}{\partial c}\right] +
\kappa \vec{k}^2 \hat{c}\right)
$$

For the PFHub equations,

$$
\frac{\partial f}{\partial c} = 2\rho
(c - c_{\alpha})(c_{\beta} - c)(c_{\alpha} + c_{\beta} - 2 c)
$$

which can be expanded out to

$$
\frac{\partial f}{\partial c} = 2\rho\left[
2 c^3 - 3(c_{\alpha} + c_{\beta}) c^2 + (c_{\alpha}^2 + 4 c_{\alpha} c_{\beta} +
c_{\beta}^2) c - (c_{\alpha}^2 c_{\beta} + c_{\alpha} c_{\beta}^2)
\right]
$$

This can be separated into a linear part:

$$
\partial_{c} f_{\mathrm{lin}} = 2\rho \left[(c_{\alpha}^2 + 4 c_{\alpha} c_{\beta} + c_{\beta}^2) c -
(c_{\alpha}^2 c_{\beta} + c_{\alpha} c_{\beta}^2)\right]
$$

and a non-linear remainder:

$$
\partial_{c} f_{\mathrm{non}} = 2\rho\left(2 c^3 - 3(c_{\alpha} + c_{\beta}) c^2\right)
$$

It's straight-forward to transform the linear expression:

$$
\mathfrak{F}\left[\partial_{c} f_{\mathrm{lin}}\right] =
2\rho \left[(c_{\alpha}^2 + 4 c_{\alpha} c_{\beta} + c_{\beta}^2) \hat{c} -
(c_{\alpha}^2 c_{\beta} + c_{\alpha} c_{\beta}^2)\right]
$$

The non-linear remainder must be evaluated in real space, then transformed into
reciprocal space, at each timestep.

A semi-implicit discretization starts with an explicit Euler form,
then assigns the linear terms to the "new" timestep. Doing so, grouping terms,
and rearranging, we arrive at the spectral discretization for this problem:

$$
\widehat{c_{t + \Delta t}} = \frac{\widehat{c_{t}} - \Delta t M \vec{k}^2 \left(
\mathfrak{F}\left[\partial_{c} f_{\mathrm{non}}\right] -
2\rho(c_{\alpha}^2 c_{\beta} + c_{\alpha} c_{\beta}^2)\right)}
{1 + \Delta t M\left[2\rho\vec{k}^2(c_{\alpha}^2 + 4 c_{\alpha} c_{\beta} +
c_{\beta}^2) + \kappa \vec{k}^4\right]}
$$

## Sweep for Non-Linearity

The non-linear term on the r.h.s. of the discretized equation of motion can
make convergence of the solution elusive. The non-linearity can be smoothed out
by sweeping the solver, rather than directly solving just once.

``` python
def step_in_time(dt):
    denom = 1 + dt * M * Ksq * (2 * ρ * (α**2 + 4 * α * β + β**2) + κ * Ksq)
    c_old = c
    c_hat_old = c_hat
    residual = 1.0

    while residual > 1e-3:
        dfdc_hat = fft2(dfdc_non(c))
        c_hat = (c_hat_old - dt * M * Ksq * dfdc_hat) / denom
        c = ifft2(c_hat)
        residual = np.linalg.norm(c - c_old)
        
    t += dt
```

## References

* _Coarsening kinetics from a variable-mobility Cahn-Hilliard equation:
  Application of a semi-implicit Fourier spectral method_,
  DOI: [10.1103/PhysRevE.60.3564](https://doi.org/10.1103/PhysRevE.60.3564)
* _Maximally fast coarsening algorithms_,
  DOI: [10.1103/PhysRevE.72.055701](https://doi.org/10.1103/PhysRevE.72.055701)
* _Controlling the accuracy of unconditionally stable algorithms in the
  Cahn-Hilliard equation_,
  DOI: [10.1103/PhysRevE.75.017702](https://doi.org/10.1103/PhysRevE.75.017702)
