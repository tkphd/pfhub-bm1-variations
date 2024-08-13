# Variations on the PFHub BM 1 Initial Conditions

Source code (Python and Jupyter) for PFHub BM 1a (spinodal decomposition with
periodic boundary conditions) using [Steppyngstounes][steppyngstounes] and a
semi-implicit spectral solver, with variations on the initial condition.

* **ori** is the "original" IC, not periodic at all at the boundaries.
* **per** is a purely periodic IC, with coefficients replaced by even multiples
  of $\pi/L$ as close to the original values as could be managed. This is
  numerically better-behaved, but produces a "boring" microstructure.
* **win** is the original IC everywhere, with [Hann windows][hann] in $x$ and
  $y$ dropped on top to produce a system that can be represented using only
  trigonometric functions. It is currently the "best" IC for spectral solvers.

The coefficients are explored in `initial_conditions.py`.
Some discussion and comparison of the initial conditions is in the `slides`
folder.

## Key Files

1. Most of the interesting mathematical details are implemented in
   [`spectral.py`](./spectral.py).
2. The initial conditions and time-stepping loop are implemented in
   [`spectral-bm1a-variations.py`](./spectral-bm1a-variations.py).
3. The FFT back-end is provided by [pyfftw][pyfftw].

## Discretization

> Thanks to Nana Ofori-Opoku (McMaster University) for fruitful
> exposition of the spectral method for Cahn-Hilliard.

Broadly, the Cahn-Hilliard equation of motion is

$$
\frac{∂ c}{∂ t} = M ∇²\left[\frac{∂ f}{∂ c} - κ ∇² c\right]
$$

Using the Fourier transform from real to reciprocal space means convolutions
(e.g., $∇ c$ and scalar multiplication) become multiplications in
reciprocal space, while exponents in real space (i.e., $c^{n\neq 1}$) become
convolutions in reciprocal space. The former simplifies life; the latter does
not. In practice, convolutions are transformed, and non-linear terms are solved
in real space and then transformed. Specifically (with Dirac's [$\delta$](https://en.wikipedia.org/wiki/Dirac_delta_function)
representing a unit impulse),

$$ \widehat{∇ c} = i\vec{k}\hat{c} $$

$$ \widehat{∇² c} = -\vec{k}² \hat{c}$$

$$ \widehat{\mathrm{const}} = \delta(\mathrm{const}) $$

Transforming the equation of motion, we have

$$
\frac{∂ \hat{c}}{∂ t} = - M \vec{k}² \left( \widehat{\frac{∂ f}{∂ c}} + κ \vec{k}² \hat{c}\right)
$$

For the PFHub equations,

$$
\frac{∂ f}{∂ c} = 2ρ (c - α)(β - c)(α + β - 2 c)
$$

which can be expanded out to

$$
\frac{∂ f}{∂ c} = 2ρ\left[2 c³ - 3(α + β) c + (α² + 4 α β + β²) c - (α² β + α β²)\right]
$$

The non-linear terms must be evaluated in real space, then transformed into
reciprocal space, at each timestep.

A semi-implicit discretization starts with an explicit Euler form,
then assigns the linear terms to the "new" timestep. Doing so, grouping terms,
and rearranging, we arrive at the spectral discretization for this problem:

$$
\widehat{c_{t + \Delta t}} = \frac{\widehat{c_{t}} - \Delta t M \vec{k}² \left(\widehat{∂_{c} f_{\mathrm{nonlin}}} - 2ρ(α² β + α β²)\right)}{1 + \Delta t M\left[2ρ\vec{k}²(α² + 4 α β + β²) + κ \vec{k}⁴\right]}
$$

## Stable Solution

Mowei Cheng published an unconditionally stable semi-implicit spectral
discretization of a simpler, but similar, model:

$$ f(φ) = ¼\left(1 - φ²\right)²,\ φ \in [-1, 1] $$

$$ \frac{∂ f}{∂ φ} = φ³ - φ $$

$$ \frac{∂ φ}{∂ τ} = ∇²\left[\frac{∂ f}{∂ φ} - γ ∇² φ\right] $$

To use the discretization, we need to transform $c$ to $φ$, $t$ to $τ$,
and $κ$ to $γ$. As our \emph{ansatz}, let's assume a linear scaling
between the field variables. Using the four known domain boundaries
(α and β for $c$, -1 and 1 for $φ$), linear interpolation yields:

$$ c(φ) = ½(β - α)(1 + φ) $$

Similarly, assume a linear temporal scaling between "our" time $t$
and Cheng's time $τ$:

$$ t = Ⲧ τ$$

From this, we can differentiate (ref: TKR6p560):

$$ ∇² c = ½(β - α) ∇²φ $$

$$ \frac{1}{ρMⲦ(β - α)²} \frac{∂ φ}{∂ τ} = ∇²\left[φ³ - φ - \frac{κ}{ρ(β - α)²} ∇² φ\right] $$

Normalizing by the coefficient of $μ(φ)$ yields

$$ γ = \frac{κ}{ρ(β - α)²} $$

$$ Ⲧ = \frac{1}{ρM(β - α)²} $$

These factors allow us to use Cheng's spectral discretization:

$$
\left[1 - Δτ k² (1 - a₁) - Δτ k⁴ γ (1 - a₂)\right] \widehat{φₙ} = \left[1 + Δτ k² a₁ - Δτ k⁴ γ a₂\right] \widehat{φₒ} - Δτ k² \widehat{φₒ³}
$$

$a₁$ and $a₂$ controls the stability and degree of implicitness.
In this model, $a₁ > 1$ and $a₂ < ½$ are unconditionally stable.

## References

* Zhu, Chen, Shen, and Tikare (1999),
  _Coarsening kinetics from a variable-mobility Cahn-Hilliard equation: Application of a semi-implicit Fourier spectral method_,
  DOI: [10.1103/PhysRevE.60.3564](https://doi.org/10.1103/PhysRevE.60.3564)
* Vollmayr-Lee and Rutenberg (2003),
  _Fast and accurate coarsening simulation with an unconditionally stable time step_,
  DOI: [10.1103/PhysRevE.68.066703](https://doi.org/10.1103/PhysRevE.68.066703)
* Cheng and Rutenberg (2005),
  _Maximally fast coarsening algorithms_,
  DOI: [10.1103/PhysRevE.72.055701](https://doi.org/10.1103/PhysRevE.72.055701)
* Cheng and Wheeler (2007),
  _Controlling the accuracy of unconditionally stable algorithms in the Cahn-Hilliard equation_,
  DOI: [10.1103/PhysRevE.75.017702](https://doi.org/10.1103/PhysRevE.75.017702)

<!-- links -->
[hann]: https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows
[pyfftw]: https://hgomersall.github.io/pyFFTW/
[steppyngstounes]: https://pages.nist.gov/steppyngstounes/en/main/index.html
