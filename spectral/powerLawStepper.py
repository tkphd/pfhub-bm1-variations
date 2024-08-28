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

    __doc__ += Stepper._stepper_test(StepperClass="PowerLawStepper")

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
        self.time = float(start)

    def _adaptStep(self):
        """Calculate next step after success

        Returns
        -------
        float
            New step.
        """
        return self.prefactor * self.time**(2 / 3)

    def succeeded(self, value):
        """Test if step was successful.

        Parameters
        ----------
        value : float, required
            Current time of the simulation.

        Returns
        -------
        bool
            Whether step was successful.  If `error` is not required,
            returns `True`.
        """
        self.time = value
        return self.stepper.succeeded(step=self)
