# PFHub BM 1 in FiPy with Steppyngstounes

Source code (Python and Jupyter) for PFHub BM 1a (periodic spinodal
decomposition) using FiPy and Steppyngstounes, with variations on the
initial condition.

* **orig** is the "original" IC, not periodic at all
* **peri** is a purely periodic IC, with coefficients
  as close to the original as could be managed
* **zany** is close to periodic, but not, to ward off
  simple periodic pattern formation

The coefficients are explored in `initial_conditions.py`.

## PyMKS

Consider using the [PyMKS Cahn-Hilliard example][pymks] to get a sense
for the spectral solver's performance?

Relevant papers:

* _Coarsening kinetics from a variable-mobility Cahn-Hilliard equation:
  Application of a semi-implicit Fourier spectral method_,
  DOI: [10.1103/PhysRevE.60.3564](https://doi.org/10.1103/PhysRevE.60.3564)
* _Maximally fast coarsening algorithms_,
  DOI: [10.1103/PhysRevE.72.055701](https://doi.org/10.1103/PhysRevE.72.055701)
* _Controlling the accuracy of unconditionally stable algorithms in the
  Cahn-Hilliard equation_,
  DOI: [10.1103/PhysRevE.75.017702](https://doi.org/10.1103/PhysRevE.75.017702)

<!-- links -->
[pymks]: https://pymks.readthedocs.io/en/stable/rst/notebooks/cahn_hilliard.html
