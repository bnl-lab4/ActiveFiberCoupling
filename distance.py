"""
Defines the `Distance` class as a general quantity with several equivalent units.
"""

from __future__ import annotations

from collections.abc import Sequence as sequence  # sorry
from typing import Sequence, SupportsFloat, Union

import sigfig

from logging_utils import get_logger

# unique logger name for this module
logger = get_logger(__name__)


def _sfround(number, sigfigs: int = 3, decimals: int = 1) -> str:
    """
    Convinience wrapper on sigfig.round for `Distance.prettyprint`.

    Uses `sigfig.round` to round a number appropriately for
    pretty printing. Numbers inside the range [1,-1] are rounded
    to `decimals` decimal places, while numbers outside that range are
    rounded to `sigfigs` significant figures. Warnings from `sigfig`
    are suppressed.

    Parameters
    ----------
    number : int, float
        The number to be rounded.
    sigfigs : int=3
        How many significant figures to round `number` to, if applicable.
    decimals : int=1
        How many decimal places to round `number` to, if applicable.

    Returns
    -------
    str
        String of the value of `number` after rounding.
    """
    if abs(number) >= 1:
        return sigfig.round(number, decimals=decimals, warn=False)
    else:
        return sigfig.round(number, sigfigs=sigfigs, warn=False)


class Distance:
    """
    A general object whose value can be in the following units:
    microns, volts, steps, fullsteps.

    Instead of restricting users to think about distances in a single unit,
    such as microns, use of `Distance` allows for distances to be input
    and output in any of the four units. The value is stored internally in
    microns. Typically, volts is used in reference to piezoelectric drivers,
    and (full)steps are used in reference to stepper motors. The micron:volt
    conversion factor comes from the ThorLabs 3-axis NanoMax flexure stage
    specifications
    (<https://www.thorlabs.com/NewGroupPage9.cfm?ObjectGroup_ID=2386>).
    The micron:(full)step conversion factor comes from the ThorLabs DRV208
    stepper motor actuator
    (<https://www.thorlabs.com/thorproduct.cfm?partnumber=DRV208>) used with
    a Pololu T834 controller (<https://www.pololu.com/product/3132>).

    Parameters
    ----------
    value : int, float
        Value of the distance.
    unit : str, default='microns'
        Unit of the given `value` (default 'microns')

    Attributes
    ----------
    microns : float
        Value in microns
    volts : float
        Value in volts
    steps : float
        Value in steps
    fullsteps : float
        Value in full steps

    Methods
    -------
    prettyprint(which=None, stacked=False)
        Print the value(s), formatted nicely and rounded correctly.

    Notes
    -----
    `Distances` can be added, subtracted, multiplied, and divided just like
    floats. Comparisons, such as ``>``, are also defined.
    """

    _MICRONS_PER_VOLT = 20 / 75
    _MICRONS_PER_FULL_STEP = 2.5
    _MICRONS_PER_STEP = _MICRONS_PER_FULL_STEP / 32
    # enforcing 32 microsteps per full step in StageAxis __init__

    def __init__(self, value: SupportsFloat, unit: str = "microns"):
        if unit == "microns":
            self._microns = float(value)
        elif unit == "volts":
            self._microns = float(value) * self._MICRONS_PER_VOLT
        elif unit == "steps":
            self._microns = float(value) * self._MICRONS_PER_STEP
        elif unit == "fullsteps":
            self._microns = float(value) * self._MICRONS_PER_FULL_STEP
        else:
            raise ValueError(
                f"Unsupported unit '{unit}':"
                + "unit must be 'microns', 'volts', 'steps', or 'fullsteps'"
            )

    ############# Operations

    def __neg__(self) -> Distance:
        """
        Negate the value.

        Returns
        -------
        `Distance`
            A new instance of the `Distance` class with an oppositely-signed value.
        """
        return Distance(-self.microns, "microns")

    def __abs__(self) -> Distance:
        """
        Take the absolute value.

        Returns
        -------
        `Distance`
            A new instance of the `Distance` class with a positively-signed value.
        """
        return Distance(abs(self.microns), "microns")

    def __add__(self, other: Distance) -> Distance:
        """
        Add with another `Distance`.

        Parameters
        ----------
        other : `Distance`
            The `Distance` to add to the current instance.

        Returns
        -------
        `Distance`
            A new `Distance` whose value is the sum.
        """
        if isinstance(other, Distance):
            new_microns = self.microns + other.microns
            return Distance(new_microns, "microns")

        return NotImplemented

    def __radd__(self, other: Distance) -> Distance:
        """
        See `__add__`.
        """
        return self.__add__(other)

    def __sub__(self, other: Distance) -> Distance:
        """
        Subtract another `Distance` from this.

        Parameters
        ----------
        other : `Distance`
            The `Distance` to subtract from the current instance.

        Returns
        -------
        `Distance`
            A new instance of the `Distance` class whose value is the difference.
        """
        if isinstance(other, Distance):
            new_microns = self.microns - other.microns
            return Distance(new_microns, "microns")

        return NotImplemented

    def __rsub__(self, other: Distance) -> Distance:
        """
        See `__sub__`.
        """
        return self.__sub__(other)

    def __mul__(self, other: SupportsFloat) -> Distance:
        """
        Multiply this object with a scalar.

        Parameters
        ----------
        other : int, float
            The scalar to multiply the current instance by.

        Returns
        -------
        `Distance`
            A new instance of the `Distance` class whose value is the product.
        """
        if isinstance(other, SupportsFloat):
            new_microns = self.microns * float(other)
            return Distance(new_microns, "microns")

        return NotImplemented

    def __rmul__(self, other: SupportsFloat) -> Distance:
        """
        See `__mul__`.
        """
        return self.__mul__(other)

    def __truediv__(self, other: SupportsFloat) -> Distance:
        """
        Divide this object by a scalar.

        Parameters
        ----------
        other : int, float
            The scalar to divide the current instance by.

        Returns
        -------
        `Distance`
            A new instance of the `Distance` class whose value is the quotient.
        """
        if isinstance(other, SupportsFloat):
            new_microns = self.microns / float(other)
            return Distance(new_microns, "microns")

        return NotImplemented

    def __rtruediv__(self, other: SupportsFloat) -> Distance:
        """
        See `__truediv__`.
        """
        return self.__truediv__(other)

    def __eq__(self, other: object) -> bool:
        """
        Compares this object to another `Distance` for equality.

        Parameters
        ----------
        other : `Distance`
            The `Distance` with which to check equality.

        Returns
        -------
        bool
            True if `Distances` are equal; False otherwise.
        """
        if isinstance(other, Distance):
            return self.microns == other.microns

        return False

    def __ne__(self, other: object) -> bool:
        """
        Compares this object to another `Distance` for inequality. See `__eq__`.
        """
        if isinstance(other, Distance):
            return self.microns != other.microns

        return False

    def __lt__(self, other: Distance) -> bool:
        """
        Checks whether this object is less than another `Distance`.

        Parameters
        ----------
        other : `Distance`
            `Distance` to compare this object to.

        Returns
        -------
        bool
            True if this object is less than other; False otherwise.
        """
        if isinstance(other, Distance):
            return self.microns < other.microns

        return NotImplemented

    def __gt__(self, other: Distance) -> bool:
        """
        Checks whether this object is greater than another `Distance`.
        See `__lt__`.
        """
        if isinstance(other, Distance):
            return self.microns > other.microns

        return NotImplemented

    def __le__(self, other: Distance) -> bool:
        """
        Checks whether this object is less than or equal to another `Distance`.
        See `__lt__` and `__eq__`.
        """
        if isinstance(other, Distance):
            return self.microns <= other.microns

        return NotImplemented

    def __ge__(self, other: Distance) -> bool:
        """
        Checks whether this object is greater than or equal to another `Distance`.
        See `__gt__` and `__eq__`.
        """
        if isinstance(other, Distance):
            return self.microns >= other.microns

        return NotImplemented

    ################# Properties

    @property
    def microns(self) -> float:
        """
        float : The distance in microns.

        Under the hood, `Distance` stores and computes everything in microns.
        """
        return self._microns

    @microns.setter
    def microns(self, value: SupportsFloat) -> None:
        self._microns = float(value)
        return

    @property
    def volts(self) -> float:
        """
        float : The distance in volts.
        """
        return self._microns / self._MICRONS_PER_VOLT

    @volts.setter
    def volts(self, value: SupportsFloat) -> None:
        self._microns = float(value) * self._MICRONS_PER_VOLT
        return

    @property
    def steps(self) -> float:
        """
        float : The distance in steps.

        There are 32 steps in 1 full step.
        """
        return self._microns / self._MICRONS_PER_STEP

    @steps.setter
    def steps(self, value: SupportsFloat) -> None:
        self._microns = float(value) * self._MICRONS_PER_STEP

    @property
    def fullsteps(self) -> float:
        """
        float : the distance in full steps.
        """
        return self._microns / self._MICRONS_PER_FULL_STEP

    @fullsteps.setter
    def fullsteps(self, value: SupportsFloat) -> None:
        self._microns = float(value) * self._MICRONS_PER_FULL_STEP

    ######### Functions

    def prettyprint(
        self, which: Union[Sequence[str], str, None] = None, stacked: bool = False
    ) -> str | list[str]:
        """
        Print the value nicely formatted and rounded.

        Prints the value of the object in one or more units. The values
        are rounded to three sigfigs or one decimal place, depending on
        the value.

        Parameters
        ----------
        which : Sequence[str], str, optional
            A single string or a sequence of strings, each of which must
            be one of the following: 'all', 'microns', 'volts',
            'steps', 'fullsteps'. The default value of ``None``
            is equivalent to 'all'.
        stacked : bool=False
            If True, all of the requested values will be printed in a single
            line. If False (default), a newline character will be inserted
            between each requested value.

        Returns
        -------
        str
            This object's value formatted in one or more units.
        """
        if isinstance(which, sequence):
            which = [s.lower() for s in which]
            if "all" in which:
                which = "all"
        if which is None or which == "all":
            which = ("microns", "volts", "steps", "fullsteps")
        if isinstance(which, str):
            which = (which,)

        output = []
        if "microns" in which:
            output.append(f"{_sfround(self.microns)} microns")
        if "volts" in which:
            output.append(f"{_sfround(self.volts)} volts")
        if "steps" in which:
            output.append(f"{_sfround(self.steps)} steps")
        if "fullsteps" in which:
            output.append(f"{_sfround(self.fullsteps)} full steps")

        output = "Distance(" + ", ".join(output) + ")"

        if stacked:
            output = output.replace(", ", ",\n" + " " * 9)

        return output
