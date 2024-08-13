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
\frac{∂ c}{∂ t} = M ∇^{2}\left[\frac{∂ f}{∂ c} - κ ∇^{2} c\right]
$$

Using the Fourier transform from real to reciprocal space means convolutions
(e.g., $∇ c$ and scalar multiplication) become multiplications in
reciprocal space, while exponents in real space (i.e., $c^{n\neq 1}$) become
convolutions in reciprocal space. The former simplifies life; the latter does
not. In practice, convolutions are transformed, and non-linear terms are solved
in real space and then transformed. Specifically (with Dirac's [$\delta$](https://en.wikipedia.org/wiki/Dirac_delta_function)
representing a unit impulse),

$$ \widehat{∇ c} = i\vec{k}\hat{c} $$

$$ \widehat{∇^{2} c} = -k^{2} \hat{c}$$

$$ \widehat{\mathrm{const}} = \delta(\mathrm{const}) $$

Transforming the equation of motion, we have

$$
\frac{∂ \hat{c}}{∂ t} = - M k^{2} \left( \widehat{\frac{∂ f}{∂ c}} + κ k^{2} \hat{c}\right)
$$

For the PFHub equations,

$$
\frac{∂ f}{∂ c} = 2ρ (c - α)(β - c)(α + β - 2 c)
$$

which can be expanded out to

$$
\frac{∂ f}{∂ c} = 2ρ\left[2 c^{3} - 3(α + β) c + (α^{2} + 4 α β + β^{2}) c - (α^{2} β + α β^{2})\right]
$$

The non-linear terms must be evaluated in real space, then transformed into
reciprocal space, at each timestep.

A semi-implicit discretization starts with an explicit Euler form,
then assigns the linear terms to the "new" timestep. Doing so, grouping terms,
and rearranging, we arrive at the spectral discretization for this problem:

$$
\widehat{c_{t + \Delta t}} = \frac{\widehat{c_{t}} - \Delta t M \vec{k}^{2} \left(\widehat{∂_{c} f_{\mathrm{nonlin}}} - 2ρ(α^{2} β + α β^{2})\right)}{1 + \Delta t M\left[2ρ\vec{k}^{2}(α^{2} + 4 α β + β^{2}) + κ \vec{k}^{4}\right]}
$$

## Stable Solution

Mowei Cheng published an unconditionally stable semi-implicit spectral
discretization of a simpler, but similar, model:

$$ f(φ) = \frac{1}{4}\left(1 - φ^{2}\right)^{2},\ φ \in [-1, 1] $$

$$ \frac{∂ f}{∂ φ} = φ^{3} - φ $$

$$ \frac{∂ φ}{∂ τ} = ∇^{2}\left[\frac{∂ f}{∂ φ} - γ ∇^{2} φ\right] $$

To use the discretization, we need to transform $c$ to $φ$, $t$ to $τ$,
and $κ$ to $γ$. As our _ansatz_, let's assume a linear scaling
between the field variables. Using the four known domain boundaries
($α$ and $β$ for $c$, -1 and 1 for $φ$), linear interpolation yields:

$$ c(φ) = \frac{1}{2}(β - α)(1 + φ) $$

Similarly, assume a linear temporal scaling between "composition" time $t$ and
"phase" time $τ$:

$$ t = Ⲧ τ$$

From this, we can differentiate (ref: TKR6p560):

$$ ∇^{2} c = \frac{1}{2}(β - α) ∇^{2}φ $$

$$ \frac{1}{ρMⲦ(β - α)^{2}} \frac{∂ φ}{∂ τ} = ∇^{2}\left[φ^{3} - φ - \frac{κ}{ρ(β - α)^{2}} ∇^{2} φ\right] $$

Normalizing by the coefficient of $μ(φ)$ yields

$$ γ = \frac{κ}{ρ(β - α)^{2}} $$

$$ Ⲧ = \frac{1}{ρM(β - α)^{2}} $$

These factors allow us to use Cheng's spectral discretization:

$$
\left[1 - \Delta τ k^{2} (1 - a_{1}) - \Delta τ k^{4} γ (1 - a_{2})\right] \widehat{φ_{\mathrm{new}}} = \left[1 + \Delta τ k^{2} a_{1} - \Delta τ k^{4} γ a_{2}\right] \widehat{φ_{\mathrm{old}}} - \Delta τ k^{2} \widehat{φ_{\mathrm{old}}^{3}}
$$

$a_{1}$ and $a_{2}$ controls the stability and degree of implicitness.
In this model, $a_{1} > 1$ and $a_{2} < \frac{1}{2}$ are unconditionally
stable; the paper recommends $a_{1} = 2$ and $a_{2} = 0$.

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
