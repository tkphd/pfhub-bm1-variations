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

$$ \mathfrak{F}\left[\nabla \phi\right] = i\vec{k} $$
$$ \mathfrak{F}\left[\nabla^2 \phi\right] = -\vec{k}^2 $$
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
\frac{\partial f}{\partial c} = 2\rho\left(
2 c^3 - 3(c_{\alpha} + c_{\beta}) c^2 + (c_{\alpha}^2 + 4 c_{\alpha} c_{\beta} +
c_{\beta}^2) c - (c_{\alpha}^2 c_{\beta} + c_{\alpha} c_{\beta}^2)
\right)
$$

This can be separated into linear and non-linear parts:

$$
\left.\frac{\partial f}{\partial c}\right|_{\mathrm{lin}} =
2\rho \left((c_{\alpha}^2 + 4 c_{\alpha} c_{\beta} + c_{\beta}^2) c -
(c_{\alpha}^2 c_{\beta} + c_{\alpha} c_{\beta}^2)
\right)
$$

$$
\left.\frac{\partial f}{\partial c}\right|_{\mathrm{non}} =
2\rho\left(2 c^3 - 3(c_{\alpha} + c_{\beta}) c^2\right)
$$

It's straight-forward to transform the linear expression:

$$
\mathfrak{F}\left[\frac{\partial f}{\partial c}\right]_{\mathrm{lin}} =
2\rho \left((c_{\alpha}^2 + 4 c_{\alpha} c_{\beta} + c_{\beta}^2) \hat{c} -
(c_{\alpha}^2 c_{\beta} + c_{\alpha} c_{\beta}^2)
\right)
$$

The non-linear remainder must be evaluated in real space, then transformed into
reciprocal space, at each timestep.

A semi-implicit discretization starts with an explicit Euler form,
then assigns the linear terms to the "new" timestep. Doing so, grouping terms,
and rearranging, we arrive at the spectral discretization for this problem:

$$
\hat{c}_{t+\Delta t} = \frac{\hat{c}_{t} - \Delta t M \vec{k}^2 \left(
\mathfrak{F}\left[\frac{\partial f}{\partial c}\right]_{\mathrm{non}} -
2\rho(c_{\alpha}^2 c_{\beta} + c_{\alpha} c_{\beta}^2)\right)}
{1 + \Delta t M\left(2\rho\vec{k}^2(c_{\alpha}^2 + 4 c_{\alpha} c_{\beta} +
c_{\beta}^2) + \kappa \vec{k}^4\right)}
$$

## References

* _Coarsening kinetics from a variable-mobility Cahn-Hilliard equation:
  Application of a semi-implicit Fourier spectral method_,
  DOI: [10.1103/PhysRevE.60.3564](https://doi.org/10.1103/PhysRevE.60.3564)
* _Maximally fast coarsening algorithms_,
  DOI: [10.1103/PhysRevE.72.055701](https://doi.org/10.1103/PhysRevE.72.055701)
* _Controlling the accuracy of unconditionally stable algorithms in the
  Cahn-Hilliard equation_,
  DOI: [10.1103/PhysRevE.75.017702](https://doi.org/10.1103/PhysRevE.75.017702)
