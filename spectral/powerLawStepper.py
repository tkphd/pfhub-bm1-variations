from __future__ import division
from __future__ import unicode_literals

from steppyngstounes.stepper import Stepper

__all__ = ["PowerLawStepper"]


class PowerLawStepper(Stepper):
    r"""Adaptive stepper that adjusts the step based on free energy density.

    Calculates a new step as

    .. math::

       \Delta = A (B \mathcal{f}^{-3})^{2/3} = A B^{2/3} / \mathcal{f}^2

    Parameters
    ----------
    start : float
        Beginning of range to step over.
    stop : float
        Finish of range to step over.
    f0 : float
        Initial energy density.
    prefactor : float
        Pre-exponential coefficient.
    minStep : float
        Smallest allowable step.
    inclusive : bool
        Whether to include an evaluation at `start` (default False)
    """

    __doc__ += Stepper._stepper_test(StepperClass="PowerLawStepper", steps=296, attempts=377)

    def __init__(self, start, stop, f0, prefactor=0.001, minStep=1e-12, inclusive=False):
        super(PowerLawStepper, self).__init__(
            start=start,
            stop=stop,
            inclusive=inclusive,
            minStep=minStep,
            size=None,
            record=False,
            limiting=False,
        )

        self.A = float(prefactor)
        self.B23 = 0.484
        self._values.append(float(f0))

    def powerlaw(self, x):
        return self.A * self.B23 / (x**2)

    def _adaptStep(self):
        """Calculate next step after success

        Returns
        -------
        float
            New step.
        """
        return self.powerlaw(self._values[-1])
