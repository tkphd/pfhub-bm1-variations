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
\frac{∂c}{∂t} = M ∇^{2}\left[\frac{∂f}{∂c} - κ ∇^{2} c\right]
$$

Using the Fourier transform from real to reciprocal space means convolutions
(e.g., $∇ c$ and scalar multiplication) become multiplications in
reciprocal space, while exponents in real space (i.e., $c^{n\neq 1}$) become
convolutions in reciprocal space. The former simplifies life; the latter does
not. In practice, convolutions are transformed, and non-linear terms are solved
in real space and then transformed. Specifically (with Dirac's [𝛿][delta]
representing a unit impulse),

$$
\widehat{∇ c} = i\vec{k}\hat{c}
$$

$$
\widehat{∇^{2} c} = -k^{2} \hat{c}
$$

$$
\widehat{\mathrm{const}} = \delta(\mathrm{const})
$$

Transforming the equation of motion, we have

$$
\frac{∂\hat{c}}{∂t} = -Mk^{2} \left(\widehat{\frac{∂f}{∂c}} + κk^{2}\hat{c}\right)
$$

For the PFHub equations,

$$
\tilde{μ}(c) = \frac{∂f}{∂c} = 2ρ (c-α)(β-c)(α-2c+β)
$$

which can be expanded out to

$$
\tilde{μ}(c) = 2ρ\left[2c^{3} - 3(α+β)c + (α^{2} + 4αβ + β^{2})c - (α^{2}β + αβ^{2})\right]
$$

The non-linear terms must be evaluated in real space, then transformed into
reciprocal space, at each timestep.

A semi-implicit discretization starts with an explicit Euler form,
then assigns the linear terms to the "new" timestep. Doing so, grouping terms,
and rearranging, we arrive at the spectral discretization for this problem:

$$
\widehat{c_{t+\Delta t}} = \frac{\widehat{c_{t}} - \Delta tMk^{2} \left(\widehat{\tilde{μ}_{\mathrm{nonlin}}} - 2ρ(α^{2}β + αβ^{2})\right)}{1 + \Delta tM\left[2ρk^{2}(α^{2} + 4αβ + β^{2}) + κk^{4}\right]}
$$

## Stable Solution

Mowei Cheng (2007) published an unconditionally stable semi-implicit spectral
discretization of a simpler, but similar, model. The benefit of this scheme is
that the timestep can be driven using a power-law relationship with error
controlled by the prefactor $A$:

$$
\Delta τ = At_{\mathrm{s}}^{⅔}
$$

where $t_{\mathrm{s}}$ is the _structural time_,

$$
t_{\mathrm{s}} = B\varepsilon^{-n}
$$

where $B = 0.286$ and $n = 3$ for conserved fields and the free energy density
$\varepsilon = \mathcal{F}/V$.

Cheng's model has fewer parameters than the benchmark; it is summarized below.

$$
f(φ) = \frac{1}{4}\left(1 - φ^{2}\right)^{2},\ φ \in [-1, 1]
$$

$$
\tilde{μ}(φ) = \frac{∂f}{∂φ} = φ^{3} - φ
$$

$$
\frac{∂φ}{∂τ} = ∇^{2}\left[\tilde{μ}(φ) - γ∇^{2} φ\right]
$$

To use the discretization, we need to transform $c$ to $φ$, $t$ to $τ$,
and $κ$ to $γ$. As our _ansatz_, let's assume a linear scaling
between the field variables. Using the four known domain boundaries
($α$ and $β$ for $c$, -1 and 1 for $φ$), linear interpolation yields:

$$
c(φ) = \frac{β - α}{2}(1 + φ)
$$

Similarly, assume a linear temporal scaling between "composition" time $t$ and
"phase" time $τ$:

$$
t = Ⲧ τ
$$

From this, we can differentiate (ref: TKR6p560):

$$
∇^{2} c = \frac{β - α}{2} ∇^{2}φ
$$

Substituting these results into the equation of motion, then normalizing by the
coefficient of $\tilde{μ}(φ)$ yields

$$
\frac{1}{ρMⲦ(β - α)^{2}} \frac{∂φ}{∂τ} = ∇^{2}\left[φ^{3} - φ - \frac{κ}{ρ(β - α)^{2}}∇^{2}φ\right]
$$

$$
γ = \frac{κ}{ρ(β - α)^{2}}
$$

$$
Ⲧ = \frac{1}{ρM(β - α)^{2}}
$$

These factors allow us to use Cheng's spectral discretization:

$$
\widehat{φ_{\mathrm{new}}} = \frac{\left[1 + \Delta τk^{2}\left(a_{1} - a_{2}k^{2}γ\right)\right] \widehat{φ_{\mathrm{old}}} - \Delta τk^{2} \widehat{φ_{\mathrm{old}}^{3}}}{1 - \Delta τk^{2} \left[1 - a_{1} + (1 - a_{2})k^{2}γ \right]}
$$

$a_{1}$ and $a_{2}$ controls the stability and degree of implicitness.
In this model, $a_{1} > 1$ and $a_{2} < \frac{1}{2}$ are unconditionally
stable; the paper recommends $a_{1} = 3$ and $a_{2} = 0$.

## References

Zhu, Chen, Shen, and Tikare (1999),
_Coarsening kinetics from a variable-mobility Cahn-Hilliard equation: Application of a semi-implicit Fourier spectral method_,
DOI: [10.1103/PhysRevE.60.3564](https://doi.org/10.1103/PhysRevE.60.3564)

Vollmayr-Lee and Rutenberg (2003),
_Fast and accurate coarsening simulation with an unconditionally stable time step_,
DOI: [10.1103/PhysRevE.68.066703](https://doi.org/10.1103/PhysRevE.68.066703)

Cheng and Rutenberg (2005),
_Maximally fast coarsening algorithms_,
DOI: [10.1103/PhysRevE.72.055701](https://doi.org/10.1103/PhysRevE.72.055701)

Cheng and Warren (2007),
_Controlling the accuracy of unconditionally stable algorithms in the Cahn-Hilliard equation_,
DOI: [10.1103/PhysRevE.75.017702](https://doi.org/10.1103/PhysRevE.75.017702)

<!-- links -->
[delta]: https://en.wikipedia.org/wiki/Dirac_delta_function
[hann]: https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows
[pyfftw]: https://hgomersall.github.io/pyFFTW/
[steppyngstounes]: https://pages.nist.gov/steppyngstounes/en/main/index.html
