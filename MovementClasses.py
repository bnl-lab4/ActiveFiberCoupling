########### TODO:
#
####################
import logging
import time
import serial
import warnings
import enum
from SensorClasses import Sensor, SensorType
from typing import Dict, List, Optional


# unique logger name for this module
log = logging.getLogger(__name__)

class MovementType(enum.Enum):
    STEPPER = "stepper"
    PIEZO = "piezo"
    GENERAL = "general"

class Distance:
    _MICRONS_PER_VOLT = 20 / 75
    _MICRONS_PER_STEP = 2.5 # I think
    def __init__(self, value, unit):
        if unit == "microns":
            self._microns = float(value)
        elif unit == "volts":
            self._microns = float(value) * self._MICRONS_PER_VOLT
        elif unit == "steps":
            self._microns = float(value) * self._MICRONS_PER_STEP
        else:
            raise ValueError("Unsupported unit: unit must be 'microns', 'volts', or 'steps'")
        
        return

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

        raise TypeError("Addition is only supported with other Distance objects")

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            new_microns = self.microns * other
            return Distance(new_microns, 'microns')

        raise TypeError("Multiplication is only supported with scalars")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            new_microns = self.microns / other
            return Distance(new_microns, 'microns')

        raise TypeError("Multiplication is only supported with scalars")

    def __rtruediv__(self, other):
        return self.__truediv__(other)

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
    

class MoveResult:
    def __init__(self, position: float, units: str, mechanism: str, centered_piezos: bool = False):
        self.position = position
        self.units = units
        self.mechanism = mechanism
        self.centered_piezos = centered_piezos
        return

    @property
    def text(self):
        return f"set {self.mechanism} to {self.position} {self.units}" +\
                    f"{'after centering piezos' if self.centered_piezos else ''}"

class StageAxis:
    
    PIEZO_LIMITS = (Distance(0, "volts"), Distance(75, "volts"))
    PIEZO_CENTER = (PIEZO_LIMITS[1] - PIEZO_LIMITS[0]) / 2

    STEPPER_LIMITS = (Distance(0, "steps"), Distance(400, "volts"))     # incorrect
    STEPPER_CENTER = (STEPPER_LIMITS[1] - STEPPER_LIMITS[0]) / 2

    def __init__(self, axis: str, piezo, stepper):
        self.axis = axis
        self.piezo = piezo
        self.stepper = stepper
        return

    def _move_piezo(self, voltage: float) -> float:
        clamped = max(0.0, min(75.0, voltage)) #piezo voltage limits
        if clamped != voltage:
            warnings.warn(f"Requested {self.axis.upper()}={voltage:.2f}V, clamped to {clamped:.2f}V")
        command = f"{self.axis.lower()}voltage={clamped}\n"
        self.piezo.read(self.piezo.in_waiting).decode("utf-8")
        self.piezo.write(command.encode())
        self.piezo.flush()

        return MoveResult(clamped, 'volts', 'piezo')

    def _move_stepper(self, steps: float) -> float:
        # send move command to stepper
        # clamping might be good if we can identify the bounds of the stepper
        # I'm assuming this is a goto function and not a moveby function
        return MoveResult(steps, 'steps', 'stepper')

    def goto(self, position: Distance, which: Optional[MovementType] = None) -> float:
        if which == MovementType.GENERAL or which is None:
            # get stepper and piezo positions as Distance objects
            stepper_position = Distance(0, 'steps')

            self.piezo.read(8)          #seems to clear better than flushing?
            self.piezo.flush()
            self.piezo.flushInput()
            self.piezo.flushOutput()
            self.piezo.write(f"{self.axis}voltage?\n".encode())
            self.piezo.readline().decode('utf-8').strip()
            piezo_position = self.piezo.read(8).decode('utf-8').strip()[2:-1] 
            piezo_position = Distance(float(piezo_position), 'volts')

            # decide whether the position can be reached with only the piezos
            if 0 < (stepper_position - movement).volts < 75:
               return self._move_piezo(position.volts)
            self._move_piezo(PIEZO_CENTER.volts)
            result = self._move_stepper((position - (piezo_position + self.PIEZO_CENTER)).steps)
            result.centered_piezos = True
            return result

        if which == MovementType.PIEZO:
            return self._move_piezo(position.volts)
        if which == MovementType.STEPPER:
            return self._move_stepper(position.steps)

        raise ValueError("which must be a MovementType enum or None.")

    def move(self, movement: Distance, which: Optional[MovementType] = None) -> float:
        if which == MovementType.GENERAL or which is None:
            # get stepper and piezo positions as Distance objects
            stepper_position = Distance(0, 'steps')

            self.piezo.read(8)          #seems to clear better than flushing?
            self.piezo.flush()
            self.piezo.flushInput()
            self.piezo.flushOutput()
            self.piezo.write(f"{self.axis}voltage?\n".encode())
            self.piezo.readline().decode('utf-8').strip()
            piezo_position = self.piezo.read(8).decode('utf-8').strip()[2:-1] 
            piezo_position = Distance(float(piezo_position), 'volts')

            axis_position = stepper_position + piezo_position

            if 0 < (axis_position + movement).volts < 75:
                return self._move_piezo((piezo_position + movement).volts)
            self._move_piezo(PIEZO_CENTER.volts)
            result = self._move_stepper((stepper_position + movement -\
                    (piezo_position - self.PIEZO_CENTER)).steps)
            result.centered_piezos = True
            return result

        if which == MovementType.PIEZO:
            return self._move_piezo((piezo_position + movement).volts)
        if which == MovementType.STEPPER:
            return self._move_stepper((stepper_position + movement).steps)

        raise ValueError("which must be a MovementType enum or None.")
            
            

class StageDevices:
    
    def __init__(self, name: str, piezo_port: str, stepper_ports: Dict[str, str], sensor: Sensor = None,
                 piezo_baud_rate: int = 115200, require_connection: bool = False):
        self.name = name
        self.sensor = sensor
        self.axes = {axis : None for axis in stepper_ports.keys()}
        self.piezo_port = piezo_port


        piezo = None
        try:
            piezo = serial.Serial(self.piezo_port, piezo_baud_rate, timeout = 1)
            log.info(f"Connected to {piezo_port} at {piezo_baud_rate} baud.")
        except serial.SerialException as e:
            if require_connection:
                raise e
            warnings.warn(f"Error opening serial port {piezo_port}: {e}")

        # loop a similar try-except over the stepper controllers
        # while also creating the axis objects
        for axis in self.axes.keys():
            try:
                stepper = stepper_ports[axis] # connect(stepper_ports[axis])
                if stepper is None:
                    log.info(f"No connection for {axis} provided")
                else:
                    log.info(f"Connected to {stepper_ports[axis]} as axis {axis}.")
            except ConnectionException as e:
                if require_connection:
                    raise e
                warnings.warn(f"Error opening stepper port {stepper_ports[axis]} as axis {axis}: {e}")
            
            self.axes[axis] = StageAxis(axis, piezo, stepper) 


        return

    def move(self, axis: str, movement: Distance, which: Optional[MovementType] = None):
        result = self.axes[axis].move(movement, which)
        log.debug(f"{self.name}, Axis {axis} :" + result.text)
        return result


    def goto(self, axis: str, position: Distance, which: Optional[MovementType] = None):
        result = self.axes[axis].goto(position, which)
        log.debug(f"{self.name}, Axis {axis} :" + result.text)
        return result

    def read(self):
        if self.sensor is None:
            warnings.warn("No sensor assigned to this stage")
            return None
        return self.sensor.read()

    def integrate(self, Texp: int, avg: bool = True):
        if self.sensor is None:
            warnings.warn("No sensor assigned to this stage")
            return None
        return self.sensor.integrate(Texp, avg)

