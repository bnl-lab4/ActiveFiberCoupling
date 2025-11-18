import time
import logging
import contextlib
import math
from typing import Dict, Optional, Tuple

from MovementClasses import MovementType, MoveResult
from Distance import Distance

# unique logger name for this module
log = logging.getLogger(__name__)

##### Constants

WAVELENGTH = 0.65 # microns

# ##############################################################################
# ### Vector Math Helper Functions
# ##############################################################################


def _vect_sub(v1: Tuple, v2: Tuple) -> Tuple:
    """Subtracts vector v2 from v1."""
    return (v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2])


def _vect_dot(v1: Tuple, v2: Tuple) -> float:
    """Calculates the dot product of two vectors."""
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


def _vect_mag_sq(v: Tuple) -> float:
    """Calculates the squared magnitude of a vector."""
    return v[0]**2 + v[1]**2 + v[2]**2

# ##############################################################################
# ### Simulation Sensor Class
# ##############################################################################


class SimulationSensor:
    """
    Simulates a sensor reading the intensity of a Gaussian beam.
    The intensity calculation depends on the position of the stage, which is
    provided by a SimulationStageDevices instance.
    """
    def __init__(self,
                 propagation_axis: str, # defaults to y
                 focal_ratio: float = 4.0,
                 beam_waist_position: Tuple[float, float, float] = (2000.0,) * 3, # microns
                 angle_of_deviation: float = 0,
                 peak_intensity: float = 1.0):
        """
        Initializes the Gaussian beam parameters.
        Args:
            propagation_axis (str): The primary axis of propagation ('x', 'y', or 'z').
            beam_waist_radius (float): The beam waist radius (w0) in microns.
            beam_waist_position (Tuple[float, float, float]): The (x, y, z) coordinates
                of the beam waist in the lab frame, in microns.
            angle_of_deviation (float): Small angle in radians to tilt the beam.
            wavelength (float): The wavelength of the light in microns.
            peak_intensity (float): The intensity at the center of the beam waist (I0).
        """
        self.propagation_axis = propagation_axis.lower()
        self.waist_pos = beam_waist_position
        self.angle = angle_of_deviation
        self.I0 = peak_intensity
        self.stage = None # This will be set by the SimulationStageDevices instance

        NA = math.sin(math.atan(1 / (2 * focal_ratio)))
        self.w0 = math.pi * WAVELENGTH / NA

        # Calculate Rayleigh range
        self.z_R = (math.pi * self.w0**2) / WAVELENGTH

        # Determine beam propagation vector based on axis and deviation
        if self.propagation_axis == 'x':
            primary_axis = (1, 0, 0)
            deviation_dir = (0, 1, 1)
        elif self.propagation_axis == 'z':
            primary_axis = (0, 0, 1)
            deviation_dir = (1, 1, 0)
        else: # Default to 'y'
            primary_axis = (0, 1, 0)
            deviation_dir = (1, 0, 1)

        # Calculate the final (unnormalized) direction vector
        tan_angle = math.tan(self.angle)
        unnorm_dir = (
            primary_axis[0] + tan_angle * deviation_dir[0],
            primary_axis[1] + tan_angle * deviation_dir[1],
            primary_axis[2] + tan_angle * deviation_dir[2]
        )

        # Normalize the direction vector
        mag = math.sqrt(_vect_mag_sq(unnorm_dir))
        self.k_beam = (unnorm_dir[0]/mag, unnorm_dir[1]/mag, unnorm_dir[2]/mag)

    def connect_to_stage(self, stage):
        """Links the sensor to a stage to get position information."""
        self.stage = stage

    def read(self) -> float:
        """
        Calculates and returns the beam intensity at the current stage position.
        """
        if self.stage is None:
            raise RuntimeError("Sensor is not connected to a stage.")

        # Get current position from the stage in microns
        pos_dict = self.stage.get_current_position()
        sensor_pos = (pos_dict['x'].microns, pos_dict['y'].microns, pos_dict['z'].microns)
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
        w_z = self.w0 * math.sqrt(1 + (z / self.z_R)**2)

        # 3. Calculate intensity I(r, z)
        intensity = self.I0 * (self.w0 / w_z)**2 * math.exp(-2 * r_sq / w_z**2)

        return intensity

    def integrate(self, Texp: int, avg: bool = True) -> float:
        """
        Simulated integration. Since time is not simulated, this just returns
        a single reading, ignoring Texp and avg.
        """
        time.sleep(0.01)
        result = self.read()
        log.debug(f"{self.stage.name} simulation sensor power: {result:.6f}")
        return result

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

# ##############################################################################
# ### Simulation Axis and Stage Classes
# ##############################################################################


class SimulationStageAxis:
    """
    Simulates a single axis of a motion stage, tracking stepper and piezo positions.
    """
    def __init__(self, axis: str, stepper_SN: str):
        self.axis = axis
        self.stepper_SN = stepper_SN
        self.stepper_position = Distance(0, 'microns')
        self.piezo_position = Distance(0, 'microns')

        # Define simulated hardware limits
        self.PIEZO_LIMITS = (Distance(0, 'volts'), Distance(75, 'volts')) # 0-20 um range
        self.PIEZO_CENTER = self.PIEZO_LIMITS[1] / 2
        self.STEPPER_LIMITS = (Distance(0, 'microns'), Distance(4000, 'microns'))
        self.STEPPER_CENTER = self.STEPPER_LIMITS[1] / 2

    def get_stepper_position(self):
        return self.stepper_position

    def get_piezo_position(self):
        return self.piezo_position

    def get_current_position(self):
        return self.piezo_position + self.stepper_position

    def _goto_piezo(self, position: Distance) -> MoveResult:
        """Sets the piezo position, clamping it within its limits."""
        clamped = max(self.PIEZO_LIMITS[0].volts,
                              min(self.PIEZO_LIMITS[1].volts, position.volts))
        if clamped != position.volts:
            log.warning(f"Cannot move {self.axis.upper()} to {clamped} because it is" +
            "outside the piezo's limits of" +
            f"({self.PIEZO_LIMITS[0].volts}, {self.PIEZO_LIMITS[1].volts}) volts")
        self.piezo_position = Distance(clamped, 'volts')
        return MoveResult(self.piezo_position, MovementType.PIEZO)

    def _goto_stepper(self, position: Distance) -> MoveResult:
        """Sets the stepper position, clamping it within its limits."""
        clamped = max(self.STEPPER_LIMITS[0].steps,
                            min(self.STEPPER_LIMITS[1].steps, position.steps))
        if clamped != position.steps:
            log.warning(f"Cannot move {self.axis.upper()} to {clamped} because it is " +
            "outside the stepper's stage limits of" +
            f"({self.PIEZO_LIMITS[0].steps}, {self.PIEZO_LIMITS[1].steps}) steps")
        self.stepper_position = Distance(clamped, 'steps')
        return MoveResult(self.stepper_position, MovementType.STEPPER)

    def goto(self, position: Distance, which: Optional[MovementType] = None) -> MoveResult:
        if which is None:
            which = MovementType.GENERAL

        if which == MovementType.PIEZO:
            return self._goto_piezo(position)
        if which == MovementType.STEPPER:
            return self._goto_stepper(position)

        if which == MovementType.GENERAL:
            stepper_position = self.get_stepper_position()

            # decide whether the position can be reached with only the piezos
            if self.PIEZO_LIMITS[0] < position - stepper_position < self.PIEZO_LIMITS[1]:
                return self._goto_piezo(position - stepper_position)
            self._goto_piezo(self.PIEZO_CENTER)
            stepper_target = position - self.PIEZO_CENTER
            result = self._goto_stepper(stepper_target)
            result.centered_piezos = True
            return result

        raise ValueError("which must be a MovementType enum or None.")

    def move(self, movement: Distance, which: Optional[MovementType] = None) -> MoveResult:
        if which is None:
            which = MovementType.GENERAL

        if which == MovementType.PIEZO:
            return self._goto_piezo(self.get_piezo_position() + movement)
        if which == MovementType.STEPPER:
            return self._goto_stepper(self.get_stepper_position() + movement)

        if which == MovementType.GENERAL:
            stepper_position = self.get_stepper_position()
            piezo_position = self.get_piezo_position()
            if self.PIEZO_LIMITS[0] < piezo_position + movement < self.PIEZO_LIMITS[1]:
                return self._goto_piezo(piezo_position + movement)
            self._goto_piezo(self.PIEZO_CENTER)
            stepper_target = stepper_position + movement +\
                        (piezo_position - self.PIEZO_CENTER)
            result = self._goto_stepper(stepper_target)
            result.centered_piezos = True
            return result

    # --- Dummy methods to match the real hardware class interface ---
    def energize(self):
        log.info(f"Axis {self.axis} stepper {self.stepper_SN} energized")
        pass

    def deenergize(self):
        log.info(f"Stepper {self.stepper_SN} deenergized")
        pass

    def home(self):
        self.stepper_position = Distance(0, 'microns')
        log.info(f"Stepper {self.stepper_SN} homing complete, " +
                    f"zeroed at lower stage limit {self.TRUE_STEPPER_LIMITS[0].prettyprint()}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class SimulationStageDevices:
    """
    Simulates the main stage controller, managing multiple axes and a sensor.
    This class mimics the public interface of the real StageDevices class.
    """
    def __init__(self, name: str, sensor: SimulationSensor = None):
        self.name = name
        self.stepper_SNs = dict(x=f'{name}_simX', y=f'{name}_simY', z=f'{name}_simZ')
        self.axes = {axis: None for axis in self.stepper_SNs.keys()}
        self.sensor = sensor
        self._exit_stack = contextlib.ExitStack()   # for context management
        # If a simulation sensor is provided, link it to this stage instance
        if self.sensor and isinstance(self.sensor, SimulationSensor):
            self.sensor.connect_to_stage(self)
            log.info(f"{self.name} connected to SimulationSensor")

        for axis, stepper_SN in self.stepper_SNs.items():
            self.axes[axis] = SimulationStageAxis(axis, stepper_SN)
            log.info(f"{self.name} established SimulationStageAxis {stepper_SN}")

    def get_piezo_position(self) -> Dict[str, Distance]:
        return {axis: stage_axis.get_piezo_position() for axis, stage_axis in self.axes.items()}

    def get_stepper_position(self) -> Dict[str, Distance]:
        return {axis: stage_axis.get_stepper_position() for axis, stage_axis in self.axes.items()}

    def get_current_position(self) -> Dict[str, Distance]:
        return {axis: stage_axis.get_current_position() for axis, stage_axis in self.axes.items()}

    def move(self, axis: str, movement: Distance, which: Optional[MovementType] = None) -> MoveResult:
        """Delegates a relative move command to the specified axis."""
        result = self.axes[axis].move(movement, which)
        log.trace(f"{self.name}, Axis {axis} :" + result.text)
        return result

    def goto(self, axis: str, position: Distance, which: Optional[MovementType] = None) -> MoveResult:
        """Delegates an absolute move command to the specified axis."""
        result = self.axes[axis].goto(position, which)
        log.trace(f"{self.name}, Axis {axis} :" + result.text)
        return result

    def read(self) -> Optional[float]:
        """Takes a reading from the attached sensor."""
        if self.sensor is None:
            log.warning("No sensor assigned to {self.name}")
            return None
        return self.sensor.read()

    def integrate(self, Texp: int, avg: bool = True) -> Optional[float]:
        """Performs an integration with the attached sensor."""
        if self.sensor is None:
            log.warning("No sensor assigned to {self.name}")
            return None
        return self.sensor.integrate(Texp, avg)

    # --- Dummy methods to match the real hardware class interface ---
    def deenergize(self, axes: Optional[str] = None):
        if axes is None or axes.lower() == 'all':
            axes = self.axes.keys()
        for axis in list(axes):
            self.axes[axis].deenergize()

    def home(self, axes: Optional[str] = None):
        if axes is None or axes.lower() == 'all':
            axes = self.axes.keys()
        for axis in list(axes):
            self.axes[axis].home()

    def energize(self, axes: Optional[str] = None):
        if axes is None or axes.lower() == 'all':
            axes = self.axes.keys()
        for axis in list(axes):
            self.axes[axis].energize()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
