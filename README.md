# Variations on the PFHub BM 1 Initial Conditions

Source code (Python and Jupyter) for PFHub BM 1a (spinodal decomposition with
periodic boundary conditions) using [Steppyngstounes][steppyngstounes] and a
semi-implicit spectral solver, with variations on the initial condition.

* **ori** is the "original" IC, not periodic at all at the boundaries.
* **per** is a purely periodic IC, with coefficients replaced by even multiples
  of $\pi/L$ as close to the original values as could be managed. This is
  numerically better-behaved, but produces a "boring" microstructure.
* **hat** is the original IC everywhere except a zone within $w$ of the
  boundary, where $c$ smoothly steps to ½. It is, unfortunately, qualitatively
  no different from the original IC in terms of spectral convergence.
* **win** is the original IC everywhere, with a [Hann window][hann] dropped on
  top to produce a system that can be represented using only trigonometric
  functions. It is currently the "best" IC for spectral solvers.

The coefficients are explored in `initial_conditions.py`.
Some discussion and comparison of the initial conditions is in the `slides`
folder.

## Key Files

1. Most of the interesting mathematical details are implemented in
   [`spectral.py`](./spectral.py).
2. The initial conditions and time-stepping loop are implemented in
   [`spectral-bm1a-variations.py`](./spectral-bm1a-variations.py).
3. The FFT back-end is provided by [mpi4py-fft][mpifftw].

## Discretization

> Thanks to Nana Ofori-Opoku (McMaster University) for fruitful
> exposition of the spectral method for Cahn-Hilliard.

Broadly, the Cahn-Hilliard equation of motion is

$$
\frac{∂ c}{∂ t} = M ∇²\left(\frac{∂ f}{∂ c} - κ ∇² c\right)
$$

Using the Fourier transform from real to reciprocal space means convolutions
(e.g., $∇ c$ and scalar multiplication) become multiplications in
reciprocal space, while exponents in real space (i.e., $c^{n\neq 1}$) become
convolutions in reciprocal space. The former simplifies life; the latter does
not. In practice, convolutions are transformed, and non-linear terms are solved
in real space and then transformed. Specifically,

$$ 𝔉\left[∇ c\right] = i\vec{k}\hat{c} $$

$$ 𝔉\left[∇² c\right] = -\vec{k}² \hat{c}$$

$$ 𝔉\left[\mathrm{const}\right] = \mathrm{const} $$

Transforming the equation of motion, we have

$$
\frac{∂ \hat{c}}{∂ t} = - M\vec{k}²
\left( 𝔉\left[\frac{∂ f}{∂ c}\right] + κ \vec{k}² \hat{c}\right)
$$

For the PFHub equations,

$$
\frac{∂ f}{∂ c} = 2ρ (c - c_{α})(c_{β} - c)(c_{α} + c_{β} - 2 c)
$$

which can be expanded out to

$$
\frac{∂ f}{∂ c} = 2ρ\left[2 c^3 - 3(c_{α} + c_{β}) c +
(c_{α}² + 4 c_{α} c_{β} + c_{β}²) c - (c_{α}² c_{β} + c_{α} c_{β}²)\right]
$$

This can be separated into a linear part:

$$
∂_{c} f_{\mathrm{linear}} = 2ρ \left[(c_{α}² + 4 c_{α} c_{β} + c_{β}²) c -
(c_{α}² c_{β} + c_{α} c_{β}²)\right]
$$

and a non-linear remainder:

$$
∂_{c} f_{\mathrm{nonlin}} = 2ρ\left(2 c^3 - 3(c_{α} + c_{β}) c²\right)
$$

It's straight-forward to transform the linear expression:

$$
𝔉\left[∂_{c} f_{\mathrm{linear}}\right] =
2ρ \left[(c_{α}² + 4 c_{α} c_{β} + c_{β}²) \hat{c}
        - (c_{α}² c_{β} + c_{α} c_{β}²)\right]
$$

The non-linear remainder must be evaluated in real space, then transformed into
reciprocal space, at each timestep.

A semi-implicit discretization starts with an explicit Euler form,
then assigns the linear terms to the "new" timestep. Doing so, grouping terms,
and rearranging, we arrive at the spectral discretization for this problem:

$$
\widehat{c_{t + \Delta t}} = \frac{\widehat{c_{t}} -
\Delta t M \vec{k}² \left(𝔉\left[∂_{c} f_{\mathrm{nonlin}}\right] -
2ρ(c_{α}² c_{β} + c_{α} c_{β}²)\right)}{1 + \Delta t M\left[2ρ\vec{k}²(c_{α}² +
4 c_{α} c_{β} + c_{β}²) + κ \vec{k}^4\right]}
$$

## Sweep for Non-Linearity

The non-linear term on the r.h.s. of the discretized equation of motion can
make convergence of the solution elusive. The non-linearity can be smoothed out
by sweeping the solver, rather than directly solving just once. Consider the
explicit (single-pass) pseudocode marching forward in time:

``` python
def march_in_time(c, dt):
    c_hat_old = FFT(c)  # "old" value in k-space
    dfdc_hat = FFT(dfdc_nonlin(c))  # non-linear piece in k-space
    numer_coeff = dt * M * Ksq  # coefficient of non-linear terms
    denom_coeff = 1 + dt * M * Ksq * (2 * ρ * (α**2 + 4 * α * β + β**2) + κ * Ksq)

    c_hat = (c_hat_old - numer_coeff * dfdc_hat) / denom_coeff

    c_new = IFFT(c_hat)  # "new" field value

    return c_new
```

The sweeping method involves inserting increasingly good "guesses" for the
argument to the non-linear piece. At first, we use the "old" value, then solve
the same set of equations using the previous round's output as the new input.
A slight tweak to this method starts with a better initial guess (h/t @reid-a):
using the values of the previous two steps, we can use the current and old
field values to extrapolate the expected new field value. This increases the
saved state of the machinery, but should produce faster convergence:

``` python
def sweep_in_less_time(c, c_old, dt):
    c_new = 2 * c - c_old    # extrapolated field value, fixed in time
    c_hat_old = FFT(c)       # "old" field in k-space
    c_hat_prev = FFT(c_old)  # "previous sweep" field in k-space
    numer_coeff = dt * M * Ksq  # coefficient of non-linear terms
    denom_coeff = 1 + dt * M * Ksq * (2 * ρ * (α**2 + 4 * α * β + β**2) + κ * Ksq)
    residual = 1.0

    while residual > 1e-3:
        dfdc_hat = fft2(dfdc_nonlin(c_new))
        c_hat = (c_hat_old - numer_coeff * dfdc_hat) / denom_coeff

        residual = np.linalg.norm(
            np.abs(c_hat_old - numer_coeff * dfdc_hat
                             - denom_coeff * c_hat_prev)).real)

        c_hat_prev[:] = c_hat_curr
        c_new[:] = IFFT(c_hat_curr)

    return c_new
```

Each sweep computes a "new" estimate of the field value using the previous
value of the non-linear terms. Think of the residual as plugging the
previous estimate of the field value in: this computes how inaccurate the
previous sweep result was. Once the loop reaches a residual below some
tolerance, further iterations are a waste of cycles: the "new" solution has
converged.

## Performance

This code is not as fast as a C-based FFTW implementation.
It marches roughly 0.01 time units per wall second, or
80 wall seconds per unit of simulation time.

## References

* _Coarsening kinetics from a variable-mobility Cahn-Hilliard equation:
  Application of a semi-implicit Fourier spectral method_,
  DOI: [10.1103/PhysRevE.60.3564](https://doi.org/10.1103/PhysRevE.60.3564)
* _Maximally fast coarsening algorithms_,
  DOI: [10.1103/PhysRevE.72.055701](https://doi.org/10.1103/PhysRevE.72.055701)
* _Controlling the accuracy of unconditionally stable algorithms in the
  Cahn-Hilliard equation_,
  DOI: [10.1103/PhysRevE.75.017702](https://doi.org/10.1103/PhysRevE.75.017702)

<!-- links -->
[hann]: https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows
[mpifftw]: https://mpi4py-fft.readthedocs.io/en/latest/
[steppyngstounes]: https://pages.nist.gov/steppyngstounes/en/main/index.html
