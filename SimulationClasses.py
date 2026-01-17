"""
Simulated stage with virtual piezos, steppers, and sensor.
"""

import contextlib
import math
from typing import Dict, List, Optional, Tuple, Union

from typing_extensions import assert_never

from distance import Distance
from LoggingUtils import get_logger
from movement_classes import MovementType, MoveResult

# unique logger name for this module
logger = get_logger(__name__)

##### Constants

WAVELENGTH = 0.65  # microns

# ##############################################################################
# ### Vector Math Helper Functions
# ##############################################################################


def _vect_sub(v1: Tuple, v2: Tuple) -> Tuple:
    """
    Subtracts vector v2 from v1.

    Parameters
    ----------
    v1, v2 : Tuple[float, float, float]
        The two 3-tuplets to be subtracted element-wise.

    Returns
    -------
    Tuple[float, float, float]
        New 3-tuplet resulting from the element-wise subraction.
    """
    return (v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2])


def _vect_dot(v1: Tuple, v2: Tuple) -> float:
    """
    Calculates the dot product of two vectors.

    Parameters
    ----------
    v1, v2 : Tuple[float, float, float]
        The two 3-tuplets to take the dot product of.

    Returns
    -------
    float
        Result of the dot product.
    """
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


def _vect_mag_sq(v: Tuple) -> float:
    """
    Calculates the squared magnitude of a vector.

    Parameters
    ----------
    v1 : Tuple[float, float, float]
        The 3-tuplet to find the squared magnitude of.

    Returns
    -------
    float
        Squared magnitude of `v`.
    """
    return v[0] ** 2 + v[1] ** 2 + v[2] ** 2


# ##############################################################################
# ### Simulation Sensor Class
# ##############################################################################


class SimulationSensor:
    r"""
    Simulates a sensor reading the intensity of a Gaussian beam.

    The intensity calculation depends on the position of the stage, which is
    provided by a `SimulationStageDevices` instance.

    Parameters
    ----------
    propagation_axis : {'x', 'y', 'z'}
        Which axis the beam propagates along.
    focal_ratio : float, default=4.0
        Focal ratio of the beam. This used to calculate the beam waist.
        The default is that of the aspheric lenses that have been used so far.
    beam_waist_position : Tuple[float, float, float], default=(2000.0, 2000.0, 2000.0)
        The position of the beam waist in microns. Defaults to the center of
        the stepper travel range.
    angle_of_deviation : float, default=0.0
        Angle between the beam and the `propagation_axis` in radians. The
        beam is rotated in the plane that passes through the
        `beam_waist_position` and is parallel to the plane defined by
        requiring the two transverse coordinates be equal. For example,
        if `propagation_axis` is 'x', `angle_of_deviation` is 0.1, and
        `beam_waist_position` is (0,0,0), then the beam is rotated by 0.1
        radians in the y=z plane about the origin. This parameter is meant
        to simulate imperfect beam angular alignment and therefore should
        be small (<~10 deg).
    peak_intensity : float, default=1.0
        The peak intensity of the beam at `beam_waist_position`.

    Attributes
    ----------
    propagation_axis : {'x', 'y', 'z'}
        Axis along which the beam primarily (see `angle_of_deviation`)
        propagates.
    waist_post : Tuple[float, float, float]
        Position of the beam waist in microns, where the origin (0,0,0) is
        located at the lower limits of the steppers.
    angle : float
        Angle between `propagation_axis` and the true beam axis.
        See `angle_of_deviation` in Parameters.
    I0 : float
        Intensity of the beam at the waist. See `beam_waist_position`.
    w0 : float
        Beam waist. See `Notes` for discussion.
    z_R : float
        Rayleigh range. See `Notes` for discussion.
    k_beam : Tuple[float, float, float]
        Unit vector pointing in the true direction of beam propagation.
    stage : `SimulationStageDevices` or None
        Simulated stage connected to this simulated sensor. The piezo and
        stepper positions of `stage` are used to determine the intensity
        the sensor is reading. Set to ``None`` if no simulated stage is
        connected.

    Methods
    -------
    read()
        Determine the beam intensity based on the stage's current position.
    integrate(exposure_time, avg=True)
        Calls `read()`. Included for general compatibility.

    Notes
    -----
    A Gaussian beam is one that has a transverse intensity profile of a
    2-dimensional Gaussian whose width changes along the propagation axis.
    A Gaussian beam is defined by its peak intensity $I_0$, waist $w_0$,
    wavelength $\lambda$, waist position $\vec{r}_0 = (x_0, y_0, z_0)$,
    and propagation direction $\vec{k} = (k_1, k_2, k_3)$. To find the the
    intensity at position $\vec{r}$, it is convinient to define
    $r_\parallel = (\vec{r} - \vec{r}_0) \cdot \vec{k}$ and $r_\perp =
    \sqrt{\|\vec{r} - \vec{r}_0 \|^2 - r_\parallel^2}$, which are the
    distances from $\vec{r}$ to $\vec{r}_0$ parallel and perpendicular to
    $\vec{k}$, respectively. Then, the beam intensity is given by
    $$I(\vec{r}) = I_0 \left(\frac{w_0}{w(r_\parallel)} \right(^2 \exp \left(
    \frac{-2 r_\perp^2}{w(r_\parallel)^2} \right)$$ where $w(r_\parallel) =
    w_0 \sqrt{ 1 + \left( r_\parallel/r_R \right)^2}$ is the $1/e^2$
    diameter of the spot and $r_R = \pi w_0^2 n/ \lambda$ is the Rayleigh
    range, with $n$ being the refractive index of the medium. For
    convenience, the beam waist $w_0$ is calculated from the provided focal
    ratio $f$. $$w_0 = \frac{\pi \lambda}{\sin \left( \arctan
    \left( 1 / 2f \right) \right)}$$ The total power contained the beam is
    $P = \frac{1}{2} \pi w_0^2 I_0$.
    """

    def __init__(
        self,
        propagation_axis: str,
        focal_ratio: float = 4.0,
        beam_waist_position: Tuple[float, float, float] = (2000.0, 2000.0, 2000.0),
        angle_of_deviation: float = 0.0,
        peak_intensity: float = 1.0,
    ):
        self.propagation_axis = propagation_axis.lower()
        self.waist_pos = beam_waist_position
        self.angle = angle_of_deviation
        self.I0 = peak_intensity
        self.stage = None  # This will be set by the SimulationStageDevices instance

        NA = math.sin(math.atan(1 / (2 * focal_ratio)))
        self.w0 = math.pi * WAVELENGTH / NA

        # Calculate Rayleigh range
        self.z_R = (math.pi * self.w0**2) / WAVELENGTH

        # Determine beam propagation vector based on axis and deviation
        if self.propagation_axis == "x":
            primary_axis = (1, 0, 0)
            deviation_dir = (0, 1, 1)
        elif self.propagation_axis == "z":
            primary_axis = (0, 0, 1)
            deviation_dir = (1, 1, 0)
        else:
            primary_axis = (0, 1, 0)
            deviation_dir = (1, 0, 1)

        # Calculate the final (unnormalized) direction vector
        tan_angle = math.tan(self.angle)
        unnorm_dir = (
            primary_axis[0] + tan_angle * deviation_dir[0],
            primary_axis[1] + tan_angle * deviation_dir[1],
            primary_axis[2] + tan_angle * deviation_dir[2],
        )

        # Normalize the direction vector
        mag = math.sqrt(_vect_mag_sq(unnorm_dir))
        self.k_beam = (unnorm_dir[0] / mag, unnorm_dir[1] / mag, unnorm_dir[2] / mag)

    def _connect_to_stage(self, stage):
        """Links the sensor to a stage to get position information."""
        self.stage = stage

    def read(self) -> float:
        """
        Calculates and returns the beam intensity at the current stage position.

        The intensity is calculated for a single point, not over the area
        of a fiber.

        Returns
        -------
        float
            Intensity at the current stage position.
        """
        if self.stage is None:
            raise RuntimeError("Sensor is not connected to a stage.")

        # Get current position from the stage in microns
        pos_dict = self.stage.get_current_position()
        sensor_pos = (
            pos_dict["x"].microns,
            pos_dict["y"].microns,
            pos_dict["z"].microns,
        )
        # Do calculation in microns, not Distance objects

        # 1. Transform coordinates to the beam's frame
        # Vector from beam waist to sensor
        V = _vect_sub(sensor_pos, self.waist_pos)

        # z: distance along the beam axis from the waist
        z = _vect_dot(V, self.k_beam)

        # r^2: squared radial distance from the beam axis
        r_sq = _vect_mag_sq(V) - z**2

        # 2. Calculate beam parameters at position z
        # w(z): beam radius at z
        w_z = self.w0 * math.sqrt(1 + (z / self.z_R) ** 2)

        # 3. Calculate intensity I(r, z)
        intensity = self.I0 * (self.w0 / w_z) ** 2 * math.exp(-2 * r_sq / w_z**2)

        return intensity

    def integrate(self, exposure_time: int, avg: bool = True) -> float:
        """
        Calculates and returns the intensity at the current stage position.

        This is included to mirror the structure of `sensor_classes.Sensor`,
        but does not behave any differently from `read`. Exposure time and
        averaging are not simulated and therefore are ignored.

        Parameters
        ----------
        exposure_time : int
            IGNORED. Included for compatibility.
        avg : bool, default=True
            IGNORED. Included for compatibility.

        Returns
        -------
        float
            Intensity at the current stage position.
        """
        if self.stage is None:
            raise RuntimeError("Sensor is not connected to a stage.")

        result = self.read()
        logger.debug(f"{self.stage.name} simulation sensor power: {result:.6f}")
        return result

    def __enter__(self):
        """
        No-op method for use with context management.

        Returns ``self`` for use in 'with... as...' statements.
        """
        pass
        return self

    def __exit__(self, _, __, ___):
        """
        No-op method for use with context management.

        Returns ``False``, letting exceptions propogate.
        """
        pass
        return False


# ##############################################################################
# ### Simulation Axis and Stage Classes
# ##############################################################################


class SimulationStageAxis:
    """
    Simulates `movement_classes.StageAxis`, tracking stepper and
    piezo positions.

    Mirrors all attributes and methods of `movement_classes.StageAxis`,
    though some are no-op or dummies.

    Parameters
    ----------
    axis : {'x', 'y', 'z'}
        Which axis this class instance will control.
    stepper_sn : str
        Identifying string of the specific stepper.

    Attributes
    ----------
    axis : {'x', 'y', 'z'}
        Which axis of the stage is controlled.
    stepper_sn : str
        Identifying string of the specific stepper.
    stepper_position : `Distance.Distance`
        Position of the stepper.
    piezo_position : `Distance.Distance`
        Position of the piezo.
    piezo_limits : 2-tuple of `Distance.Distance`
        Upper and lower limits of the piezo travel range.
    piezo_center : `Distance.Distance`
        Center of the piezo travel range.
    stepper_limits : 2-tuple of `Distance.Distance`
        Upper and lower limits of the stepper travel range.
    stepper_center : `Distance.Distance`
        Center of the stepper travel range.

    Methods
    -------
    get_stepper_position()
        Get the current stepper position.
    get_piezo_position()
        Get the current piezo position.
    get_current_position()
        Get the combined position of the stepper and piezo.
    goto(position, which=None)
        Move the stepper and/or piezo to the desired position. ``Which``
        determines the movement type, which defaults to ``GENERAL``.
    move(movement, which=None)
        Move the stepper and/or piezo by the desired distance. ``Which``
        determines the movement type, which defaults to ``GENERAL``.
    home()
        Set the stepper position to 0 (microns).
    energize()
        No-op dummy method.
    deenergize()
        No-op dummy method.
    """

    def __init__(self, axis: str, stepper_sn: str):
        self.axis = axis
        self.stepper_sn = stepper_sn
        self.stepper_position = Distance(0, "microns")
        self.piezo_position = Distance(0, "microns")

        # Define simulated hardware limits
        self.piezo_limits = (
            Distance(0, "volts"),
            Distance(75, "volts"),
        )  # 0-20 um range
        self.piezo_center = self.piezo_limits[1] / 2
        self.stepper_limits = (Distance(0, "microns"), Distance(4000, "microns"))
        self.stepper_center = self.stepper_limits[1] / 2

    def get_stepper_position(self):
        """Get the current stepper position."""
        return self.stepper_position

    def get_piezo_position(self):
        """Get the current piezo position."""
        return self.piezo_position

    def get_current_position(self):
        """Get the current combined stage position."""
        return self.piezo_position + self.stepper_position

    def _goto_piezo(self, position: Distance) -> MoveResult:
        """
        Set the piezo to the desired position, if able.

        Parameters
        ----------
        position : `Distance.Distance`
            The position to set the piezo to. If this is outside the
            travel limits of the piezo, it will be set to the nearest
            limit.

        Returns
        -------
        `MoveResult`
            Info to be logged regarding the movement.
        """
        clamped = max(
            self.piezo_limits[0].volts, min(self.piezo_limits[1].volts, position.volts)
        )
        if clamped != position.volts:
            logger.warning(
                f"Cannot move {self.axis.upper()} to {clamped} because it is"
                + "outside the piezo's limits of"
                + f"({self.piezo_limits[0].volts}, {self.piezo_limits[1].volts}) volts"
            )
        self.piezo_position = Distance(clamped, "volts")
        return MoveResult(self.piezo_position, MovementType.PIEZO)

    def _goto_stepper(self, position: Distance) -> MoveResult:
        """
        Set the stepper to the desired position.

        Parameters
        ----------
        position : `Distance.Distance`
            The position to set the stepper to. If this is outside
            the travel limits of the stage, it will be set to the
            nearest limit.

        Returns
        -------
        `MoveResult`
            Info to be logged regarding the movement.
        """

        clamped = max(
            self.stepper_limits[0].steps,
            min(self.stepper_limits[1].steps, position.steps),
        )
        if clamped != position.steps:
            logger.warning(
                f"Cannot move {self.axis.upper()} to {clamped} because it is "
                + "outside the stepper's stage limits of"
                + f"({self.piezo_limits[0].steps}, {self.piezo_limits[1].steps}) steps"
            )
        self.stepper_position = Distance(clamped, "steps")
        return MoveResult(self.stepper_position, MovementType.STEPPER)

    def goto(
        self, position: Distance, which: Optional[MovementType] = None
    ) -> MoveResult:
        """
        Move stepper and/or piezo to the desired position.

        If `which` is ``PIEZO``, the piezo will be set to `position`, taking
        ``piezo_limits[0]`` to be zero. If `which` is ``STEPPER``, the
        stepper will be set to `position`, taking ``stepper_limits[0]`` to
        be zero.

        If `which` is ``GENERAL``, then ``stepper_limits[0]`` is
        taken to be zero. Then, if `position` is accessible with just the
        piezo, then only the piezo will be used. If the piezo cannot reach
        `position`, then the piezo is centered in its range and the
        stepper moves so that the stage itself will have moved to `position`.

        Parameters
        ----------
        position : `Distance.Distance`
            Desired position to move to.
        which : `MovementType`, optional
            Which of the movement methods to use. ``None`` defaults
            to ``GENERAL``.

        Returns
        -------
        `MoveResult`
            Info to be logged regarding the movement.
        """
        if which is None:
            which = MovementType.GENERAL

        if which == MovementType.PIEZO:
            return self._goto_piezo(position)
        if which == MovementType.STEPPER:
            return self._goto_stepper(position)

        if which == MovementType.GENERAL:
            stepper_position = self.get_stepper_position()

            # decide whether the position can be reached with only the piezos
            if (
                self.piezo_limits[0]
                < position - stepper_position
                < self.piezo_limits[1]
            ):
                return self._goto_piezo(position - stepper_position)
            self._goto_piezo(self.piezo_center)
            stepper_target = position - self.piezo_center
            result = self._goto_stepper(stepper_target)
            result.centered_piezos = True
            return result

        assert_never(which)

    def move(
        self, movement: Distance, which: Optional[MovementType] = None
    ) -> MoveResult:
        """
        Move the stepper and/or piezo by the desired distance.

        If `which` is ``PIEZO``, the piezo will be moved by `movement`.
        If `which` is ``STEPPER``, the stepper will be moved by `movement`.
        If `which` is ``GENERAL`` and `movement` can be accomplished with
        just the piezo, then only the piezo will be moved. Otherwise, the
        piezo will be set to the center of its range and the stepper will
        move so that the stage itself will have moved by `movement`.

        Parameters
        ----------
        movement : `Distance.Distance`
            Desired distance to move.
        which : `MovementType`, optional
            Which of the movement methods to use. ``None`` defaults
            to ``GENERAL``.

        Returns
        -------
        `MoveResult`
            Info to be logged regarding the movement.
        """

        if which is None:
            which = MovementType.GENERAL

        if which == MovementType.PIEZO:
            return self._goto_piezo(self.get_piezo_position + movement)
        if which == MovementType.STEPPER:
            return self._goto_stepper(self.stepper_position + movement)

        if which == MovementType.GENERAL:
            stepper_position = self.get_stepper_position()
            piezo_position = self.get_piezo_position()
            if self.piezo_limits[0] < piezo_position + movement < self.piezo_limits[1]:
                return self._goto_piezo(piezo_position + movement)
            self._goto_piezo(self.piezo_center)
            stepper_target = (
                stepper_position + movement + (piezo_position - self.piezo_center)
            )
            result = self._goto_stepper(stepper_target)
            result.centered_piezos = True
            return result

    # --- Dummy methods to match the real hardware class interface ---
    def energize(self):
        """No-op dummy method."""
        logger.info(f"Axis {self.axis} stepper {self.stepper_sn} energized")
        pass

    def deenergize(self):
        """No-op dummy method."""
        logger.info(f"Stepper {self.stepper_sn} deenergized")
        pass

    def home(self):
        """Set the stepper position to 0 (microns)."""
        self.stepper_position = Distance(0, "microns")
        logger.info(
            f"Stepper {self.stepper_sn} homing complete, "
            + f"zeroed at lower stage limit {self.stepper_limits[0].prettyprint()}"
        )

    def __enter__(self):
        """
        No-op method for use with context management.

        Returns ``self`` for use in 'with... as...' statements.
        """
        return self

    def __exit__(self, _, __, ___):
        """
        No-op method for use with context management.

        Returns ``False``, letting exceptions propogate.
        """
        return False


class SimulationStageDevices:
    """
    Simulates the main stage controller, managing multiple axes and a sensor.
    This class mimics the public interface of the real StageDevices class.

    Holds three child `SimulationStageAxis` classes and a
    `SimulationSensor` class.

    Parameters
    ----------
    name : str
        Name of the simulated stage.
    sensor : `SimulationSensor`, optional
        Associated `SimulationSensor` instance for calculating intensity.

    Attributes
    ----------
    name : str
        Name of the simulated stage.
    sensor : SimulationSensor
        Simulated sensor used for this simulated stage.
    axes : Dict[str, `StageAxis` or ``None``]
        Dictionary of stage axes, with the axis names as keys and
        `SimulationStageAxis` classes as values.
    stepper_sns : Dict[{'x', 'y', 'z'}, str]
        Dummy "serial numbers" for the steppers.

    Methods
    -------
    get_piezo_position()
        Get dictionary of piezo position.
    get_stepper_position()
        Get dictionary of stepper position
    get_current_position()
        Get dictionary of axis positions.
    move(axis, movement, which=None)
        Move an axis by some amount using its stepper and/or piezo.
    goto(axis, position, which=None)
        Move an axis to a position using its stepper and/or piezo.
    read()
        "Read" the simulated sensor, if one is defined.
    integrate(exposure_time, avg=True)
       Dummy method calling `read`. Parameters are ignored.
    home(axes=None)
        Call `home` on one or more axes.
    energize(axes=None)
        Call `energize` (no-op dummy) on one or more axes.
    deenergize(axes=None)
        Call `deenergize` (no-op dummy) on one or more axes.

    Notes
    -----
    For control of real, physical stages, see `movement_classes.py`.
    """

    def __init__(self, name: str, sensor: Optional[SimulationSensor] = None):
        self.name = name
        self.stepper_sns = dict(x=f"{name}_simX", y=f"{name}_simY", z=f"{name}_simZ")
        self.axes: Dict[str, SimulationStageAxis] = {}
        self.sensor = sensor
        self._exit_stack = contextlib.ExitStack()  # for context management
        # If a simulation sensor is provided, link it to this stage instance
        if self.sensor and isinstance(self.sensor, SimulationSensor):
            self.sensor._connect_to_stage(self)
            logger.info(f"{self.name} connected to SimulationSensor")

        for axis, stepper_sn in self.stepper_sns.items():
            self.axes[axis] = SimulationStageAxis(axis, stepper_sn)
            logger.info(f"{self.name} established SimulationStageAxis {stepper_sn}")

    def get_piezo_position(self) -> Dict[str, Distance]:
        """Returns axis-keyed dictionary of piezo positions."""
        return {
            axis: stage_axis.get_piezo_position()
            for axis, stage_axis in self.axes.items()
        }

    def get_stepper_position(self) -> Dict[str, Distance]:
        """Returns axis-keyed dictionary of stepper positions."""
        return {
            axis: stage_axis.get_stepper_position()
            for axis, stage_axis in self.axes.items()
        }

    def get_current_position(self) -> Dict[str, Distance]:
        """Returns axis-keyed dictionary of axis positions."""
        return {
            axis: stage_axis.get_current_position()
            for axis, stage_axis in self.axes.items()
        }

    def move(
        self, axis: str, movement: Distance, which: Optional[MovementType] = None
    ) -> MoveResult:
        """
        Move `axis` by `movement` using its stepper and/or piezo.

        Parameters
        ----------
        axis : {'x', 'y', 'z'}
            Which axis to move.
        movement : `Distance.Distance`
            What distance to move the given axis.
        which : `MovementType`, optional
            Which movement type to use. See `SimulationStageAxis.move` for
            more information on the behaviour of the movement types.

        Returns
        -------
        `MovementResult`
            Info to be logged regarding the movement.
        """
        result = self.axes[axis].move(movement, which)
        logger.trace(f"{self.name}, Axis {axis} :" + result.text)
        return result

    def goto(
        self, axis: str, position: Distance, which: Optional[MovementType] = None
    ) -> MoveResult:
        """
        Move `axis` to `position` using its stepper and/or piezo.

        Parameters
        ----------
        axis : {'x', 'y', 'z'}
            Which axis to move.
        position : `Distance.Distance`
            What position to move the given axis to.
        which : `MovementType`, optional
            Which movement type to use. See `SimulationStageAxis.move` for
            more information on the behaviour of the movement types.

        Returns
        -------
        `MovementResult`
            Info to be logged regarding the movement.
        """
        result = self.axes[axis].goto(position, which)
        logger.trace(f"{self.name}, Axis {axis} :" + result.text)
        return result

    def read(self) -> Optional[float]:
        """Takes a reading from the attached sensor, if it exists."""
        if self.sensor is None:
            logger.warning("No sensor assigned to {self.name}")
            return None
        return self.sensor.read()

    def integrate(self, exposure_time: int, avg: bool = True) -> Optional[float]:
        """Performs an integration with the attached sensor, if it exists."""
        if self.sensor is None:
            logger.warning("No sensor assigned to {self.name}")
            return None
        return self.sensor.integrate(exposure_time, avg)

    # --- Dummy methods to match the real hardware class interface ---
    def deenergize(self, axes: Union[str, List[str]] = "all"):
        """
        Call `deenergize` (no-op dummy) on one or more axes.

        Parameters
        ----------
        axes : {'x', 'y', 'z', None}, optional
            Which axes to call on.
        """
        if str([c.lower() for c in axes]) == "all":
            axes = list(self.axes.keys())
        for axis in list(axes):
            self.axes[axis].deenergize()

    def home(self, axes: Union[str, List[str]] = "all"):
        """
        Call `home` on one or more axes.

        Parameters
        ----------
        axes : {'x', 'y', 'z', None}, optional
            Which axes to home.
        """
        if str([c.lower() for c in axes]) == "all":
            axes = list(self.axes.keys())
        for axis in list(axes):
            self.axes[axis].home()

    def energize(self, axes: Union[str, List[str]] = "all"):
        """
        Call `energize` (no-op dummy) on one or more axes.

        Parameters
        ----------
        axes : {'x', 'y', 'z', None}, optional
            Which axes to call on.
        """
        if str([c.lower() for c in axes]) == "all":
            axes = list(self.axes.keys())
        for axis in axes:
            self.axes[axis].energize()

    def __enter__(self):
        """
        No-op method for use with context management.

        Returns ``self`` for use in 'with... as...' statements.
        """
        return self

    def __exit__(self, _, __, ___):
        """
        No-op method for use with context management.

        Returns ``False``, letting exceptions propogate.
        """
        return False
