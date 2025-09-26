import math
from typing import Dict, Optional, Tuple

# These classes are defined in the provided 'MovementClasses.py' file.
# They must be available in the same directory for this code to run.
from MovementClasses import Distance, MovementType, MoveResult

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
                 propagation_axis: str,
                 beam_waist_radius: float,
                 beam_waist_position: Tuple[float, float, float],
                 angle_of_deviation: float,
                 wavelength: float = 0.633,
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
        self.w0 = beam_waist_radius
        self.waist_pos = beam_waist_position
        self.angle = angle_of_deviation
        self.I0 = peak_intensity
        self.stage = None # This will be set by the SimulationStageDevices instance

        # Calculate Rayleigh range
        self.z_R = (math.pi * self.w0**2) / wavelength

        # Determine beam propagation vector based on axis and deviation
        if self.propagation_axis == 'x':
            primary_axis = (1, 0, 0)
            deviation_dir = (0, 1, 1)
        elif self.propagation_axis == 'y':
            primary_axis = (0, 1, 0)
            deviation_dir = (1, 0, 1)
        else: # Default to 'z'
            primary_axis = (0, 0, 1)
            deviation_dir = (1, 1, 0)

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
        sensor_pos = (pos_dict.get('x', 0.0), pos_dict.get('y', 0.0), pos_dict.get('z', 0.0))

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

        # Photodiode returns a negative voltage proportional to intensity
        return -intensity

    def integrate(self, Texp: int, avg: bool = True) -> float:
        """
        Simulated integration. Since time is not simulated, this just returns
        a single reading, ignoring Texp and avg.
        """
        return self.read()

# ##############################################################################
# ### Simulation Axis and Stage Classes
# ##############################################################################


class SimulationStageAxis:
    """
    Simulates a single axis of a motion stage, tracking stepper and piezo positions.
    """
    def __init__(self, axis: str):
        self.axis = axis
        self.stepper_position = Distance(0, 'microns')
        self.piezo_position = Distance(0, 'microns')

        # Define simulated hardware limits
        self.PIEZO_LIMITS = (Distance(0, 'microns'), Distance(20, 'microns')) # 0-20 um range
        self.PIEZO_CENTER = self.PIEZO_LIMITS[1] / 2
        self.STEPPER_LIMITS = (Distance(0, 'steps'), Distance(128000, 'steps')) # ~4000 um range

    def get_position(self) -> Distance:
        """Returns the total combined position of the axis."""
        return self.stepper_position + self.piezo_position

    def _goto_piezo(self, position: Distance) -> MoveResult:
        """Sets the piezo position, clamping it within its limits."""
        clamped_microns = max(self.PIEZO_LIMITS[0].microns, min(self.PIEZO_LIMITS[1].microns, position.microns))
        self.piezo_position = Distance(clamped_microns, 'microns')
        return MoveResult(self.piezo_position, MovementType.PIEZO)

    def _goto_stepper(self, position: Distance) -> MoveResult:
        """Sets the stepper position, clamping it within its limits."""
        clamped_steps = max(self.STEPPER_LIMITS[0].steps, min(self.STEPPER_LIMITS[1].steps, position.steps))
        self.stepper_position = Distance(clamped_steps, 'steps')
        return MoveResult(self.stepper_position, MovementType.STEPPER)

    def goto(self, position: Distance, which: Optional[MovementType] = None) -> MoveResult:
        """Moves the axis to an absolute position."""
        if which == MovementType.PIEZO:
            return self._goto_piezo(position)
        if which == MovementType.STEPPER:
            return self._goto_stepper(position)

        # Default "general" movement: center the piezo and move the stepper
        self._goto_piezo(self.PIEZO_CENTER)
        stepper_target = position - self.PIEZO_CENTER
        result = self._goto_stepper(stepper_target)
        result.centered_piezos = True
        return result

    def move(self, movement: Distance, which: Optional[MovementType] = None) -> MoveResult:
        """Moves the axis by a relative amount."""
        target_pos = self.get_position() + movement
        return self.goto(target_pos, which)

    # --- Dummy methods to match the real hardware class interface ---
    def energize(self):
        pass

    def deenergize(self):
        pass

    def home(self):
        self.stepper_position = Distance(0, 'microns')
        self.piezo_position = Distance(0, 'microns')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class SimulationStageDevices:
    """
    Simulates the main stage controller, managing multiple axes and a sensor.
    This class mimics the public interface of the real StageDevices class.
    """
    def __init__(self, name: str, stepper_SNs: Dict[str, str], sensor: SimulationSensor = None):
        self.name = name
        self.axes = {axis: SimulationStageAxis(axis) for axis in stepper_SNs.keys()}
        self.sensor = sensor
        # If a simulation sensor is provided, link it to this stage instance
        if self.sensor and isinstance(self.sensor, SimulationSensor):
            self.sensor.connect_to_stage(self)

    def get_current_position(self) -> Dict[str, float]:
        """Returns a dictionary of the current positions of all axes in microns."""
        return {axis: stage_axis.get_position().microns for axis, stage_axis in self.axes.items()}

    def move(self, axis: str, movement: Distance, which: Optional[MovementType] = None) -> MoveResult:
        """Delegates a relative move command to the specified axis."""
        return self.axes[axis].move(movement, which)

    def goto(self, axis: str, position: Distance, which: Optional[MovementType] = None) -> MoveResult:
        """Delegates an absolute move command to the specified axis."""
        return self.axes[axis].goto(position, which)

    def read(self) -> Optional[float]:
        """Takes a reading from the attached sensor."""
        if self.sensor is None:
            return None
        return self.sensor.read()

    def integrate(self, Texp: int, avg: bool = True) -> Optional[float]:
        """Performs an integration with the attached sensor."""
        if self.sensor is None:
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
