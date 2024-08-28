from __future__ import division
from __future__ import unicode_literals

from steppyngstounes.stepper import Stepper

__all__ = ["PowerLawStepper"]


class PowerLawStepper(Stepper):
    r"""Adaptive stepper that adjusts the step based on free energy density.

    Calculates a new step as

    .. math::

       \Delta = A t^{2/3}

    Parameters
    ----------
    start : float
        Beginning of range to step over.
    stop : float
        Finish of range to step over.
    prefactor : float
        Pre-exponential coefficient.
    minStep : float
        Smallest allowable step.
    inclusive : bool
        Whether to include an evaluation at `start` (default False)
    """

    __doc__ += Stepper._stepper_test(StepperClass="PowerLawStepper", steps=296, attempts=377)

    def __init__(self, start, stop, prefactor=0.001, minStep=2**-20, inclusive=False):
        super(PowerLawStepper, self).__init__(
            start=start,
            stop=stop,
            inclusive=inclusive,
            minStep=minStep,
            size=None,
            record=False,
            limiting=False,
        )

        self.prefactor = float(prefactor)
        self._values.append(float(start))

    def _adaptStep(self):
        """Calculate next step after success

        Returns
        -------
        float
            New step.
        """
        time = self._values[-1]
        return self.prefactor * time**(2 / 3)
