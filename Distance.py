import sigfig
from typing import Union, Sequence
from numbers import Real


def sfround(number, sigfigs = 3, decimals = 1):
    if abs(number) >= 1:
        return sigfig.round(number, decimals=decimals, warn=False)
    else:
        return sigfig.round(number, sigfigs=sigfigs, warn=False)


class Distance:
    _MICRONS_PER_VOLT = 20 / 75
    _MICRONS_PER_FULL_STEP = 2.5     # I think
    _MICRONS_PER_STEP = _MICRONS_PER_FULL_STEP / 32
    # enforcing 32 microsteps per full step in StageAxis __init__

    def __init__(self, value: Union[int, float], unit: str = "microns"):
        if unit == "microns":
            self._microns = float(value)
        elif unit == "volts":
            self._microns = float(value) * self._MICRONS_PER_VOLT
        elif unit == "steps":
            self._microns = float(value) * self._MICRONS_PER_STEP
        elif unit == 'fullsteps':
            self._microns = float(value) * self._MICRONS_PER_FULL_STEP
        else:
            raise ValueError("Unsupported unit: unit must be 'microns', 'volts', 'steps', or 'fullsteps'")

############# Operations

    def __neg__(self):
        return Distance(-self.microns, 'microns')

    def __abs__(self):
        return Distance(abs(self.microns), 'microns')

    def __add__(self, other):
        if isinstance(other, Distance):
            new_microns = self.microns + other.microns
            return Distance(new_microns, 'microns')

        raise TypeError("Addition is only supported with other Distance objects")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Distance):
            new_microns = self.microns - other.microns
            return Distance(new_microns, 'microns')

        raise TypeError("Subtraction is only supported with other Distance objects")

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        if isinstance(other, Real):
            new_microns = self.microns * other
            return Distance(new_microns, 'microns')

        raise TypeError("Multiplication is only supported with scalars")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Real):
            new_microns = self.microns / other
            return Distance(new_microns, 'microns')

        raise TypeError("Division is only supported with scalars")

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def __eq__(self, other):
        if isinstance(other, Distance):
            return self.microns == other.microns

        raise TypeError("Comparison only supported between Distance objects")

    def __ne__(self, other):
        if isinstance(other, Distance):
            return self.microns != other.microns

        raise TypeError("Comparison only supported between Distance objects")

    def __lt__(self, other):
        if isinstance(other, Distance):
            return self.microns < other.microns

        raise TypeError("Comparison only supported between Distance objects")

    def __gt__(self, other):
        if isinstance(other, Distance):
            return self.microns > other.microns

        raise TypeError("Comparison only supported between Distance objects")

    def __le__(self, other):
        if isinstance(other, Distance):
            return self.microns <= other.microns

        raise TypeError("Comparison only supported between Distance objects")

    def __ge__(self, other):
        if isinstance(other, Distance):
            return self.microns >= other.microns

        raise TypeError("Comparison only supported between Distance objects")

################# Properties

    @property
    def microns(self):
        return self._microns

    @microns.setter
    def microns(self, value):
        self._microns = float(value)
        return

    @property
    def volts(self):
        return self._microns / self._MICRONS_PER_VOLT

    @volts.setter
    def volts(self, value):
        self._microns = float(value) * self._MICRONS_PER_VOLT
        return

    @property
    def steps(self):
        return self._microns / self._MICRONS_PER_STEP

    @steps.setter
    def steps(self, value):
        self._microns = float(value) * self._MICRONS_PER_STEP

    @property
    def fullsteps(self):
        return self._microns / self._MICRONS_PER_FULL_STEP

    @fullsteps.setter
    def fullsteps(self, value):
        self._microns = float(value) * self._MICRONS_PER_FULL_STEP

######### Functions

    def prettyprint(self, which: Union[Sequence[str], str, None] = None,
                    stacked: bool = False):
        if which is None or which.lower() == 'all':
            which = ('microns', 'volts', 'steps', 'fullsteps')
        if isinstance(which, str):
            which = (which,)

        output = []
        if 'microns' in which:
            output.append(f'{sfround(self.microns)} microns')
        if 'volts' in which:
            output.append(f'{sfround(self.volts)} volts')
        if 'steps' in which:
            output.append(f'{sfround(self.steps)} steps')
        if 'fullsteps' in which:
            output.append(f'{sfround(self.fullsteps)} full steps')

        output = 'Distance(' + ', '.join(output) + ')'

        if stacked:
            output = output.replace(', ', ',\n' + ' ' * 9)

        return output
