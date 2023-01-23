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

<!-- links -->
[pymks]: https://pymks.readthedocs.io/en/stable/rst/notebooks/cahn_hilliard.html
