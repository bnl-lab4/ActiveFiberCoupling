########### TODO:
# each axis gets a class, then each stage gets a parent class
####################
import logging
import time
import serial
import warnings
import enum
from typing import Dict, List, Optional


# unique logger name for this module
log = logging.getLogger(__name__)

class MovementType(enum.Enum):
    STEPPER = "stepper"
    PIEZO = "piezo"
    GENERAL = "general"

class Position:
    MICRONS_PER_VOLT = 20 / 75
    MICRONS_PER_STEP = 2.5 # I think

    def __init__(self, value, unit):
        MICRONS_PER_VOLT = 20 / 75
        MICRONS_PER_STEP = 2.5 # I think

        if unit == "microns":
            self._microns = float(value)
        elif unit == "volts":
            self._microns = float(value) * MICRONS_PER_VOLT
        elif unit == "steps":
            self._microns = float(value) * MICRONS_PER_STEP
        else:
            raise ValueError("Unsupported unit: unit must be 'microns', 'volts', or 'steps'")
        
        return

    def __add__(self, other):
        if isinstance(other, Position):
            new_microns = self.microns + other.microns
            return Position(new_microns, 'microns')

        raise TypeError("Addition is only supported with other Position objects")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Position):
            new_microns = self.microns - other.microns
            return Position(new_microns, 'microns')

        raise TypeError("Addition is only supported with other Position objects")

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            new_microns = self.microns * other
            return Position(new_microns, 'microns')

        raise TypeError("Multiplication is only supported with scalars")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        if isinstance(other, (int, float)):
            new_microns = self.microns / other
            return Position(new_microns, 'microns')

        raise TypeError("Multiplication is only supported with scalars")

    def __rdiv__(self, other):
        return self.__div__(other)

    @property
    def microns(self):
        return self._microns

    @microns.setter
    def microns(self, value):
        self._microns = value
        return

    @property
    def volts(self):
        return self._microns / self.MICRONS_PER_VOLT

    @volts.setter
    def volts(self, value):
        self._microns = value * self.MICRONS_PER_VOLT
        return

    @property
    def steps(self):
        return self._microns / self.MICRONS_PER_STEP

    @steps.setter
    def steps(self, value):
        self._microns = value * self.MICRONS_PER_STEP
    

class MoveResult:
    def __init__(self, position: float, units: str, mechanism: str, centered_piezos: bool = False):
        self.position = position
        self.units = units
        self.mechanism = mechanism
        self.centered_piezos = centered_piezos
        return

    @property
    def text(self):
        return f"set {result.mechanism} to {result.position} {result.units}" +\
                    f"{'after centering piezos' if result.centered_piezos else ''}")


class StageAxis:
    
    CENTER = Position(37.5, "volts")

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

    def goto(self, position: Position, which: Optional[MovementType] = None) -> float:
        # get stepper and piezo positions as Position objects
        stepper_position = Position(0, 'steps')

        self.piezo.write(f"{self.axis}voltage?\n".encode())
        self.piezo.readline().decode('utf-8').strip()
        piezo_position = self.piezo.read(8).decode('utf-8').strip()[2:-1] 
        piezo_position = Position(float(piezo_position), 'volts')

        if which == MovementType.GENERAL or which is None:
            # decide whether the position can be reached with only the piezos
            if 0 < (stepper_position - movement).volts < 75:
               return self._move_piezo(position.volts)
            self._move_piezo(self.CENTER.volts)
            result = self._move_stepper((position - (piezo_position + self.CENTER)).steps)
            result.centered_piezos = True
            return result

        if which == MovementType.PIEZO:
            return self._move_piezo(position.volts)
        if which == MovementType.STEPPER:
            return self._move_stepper(position.steps)

    def move(self, movement: Position, which: Optional[MovementType] = None) -> float:
        # get stepper and piezo positions as Position objects
        stepper_position = Position(0, 'steps')

        self.piezo.write(f"{self.axis}voltage?\n".encode())
        self.piezo.readline().decode('utf-8').strip()
        piezo_position = self.piezo.read(8).decode('utf-8').strip()[2:-1] 
        piezo_position = Position(float(piezo_position), 'volts')

        axis_position = stepper_position + piezo_position

        if which == MovementType.GENERAL or which is None:
            if 0 < (axis_position + movement).volts < 75:
                return self._move_piezo((piezo_position + movement).volts)
            self._move_piezo(self.CENTER.volts)
            result = self._move_stepper((stepper_position + movement -\
                    (piezo_position - self.CENTER)).steps)
            result.centered_piezos = True
            return result

        if which == MovementType.PIEZO:
            return self._move_piezo((piezo_position + movement).volts)
        if which == MovementType.STEPPER:
            return self._move_stepper((stepper_position + movement).steps)
            
            

class StageDevices:
    
    def __init__(self, name: str, piezo_port: str, stepper_ports: Dict[str, str],
                 piezo_baud_rate: int = 115200, require_connection: bool = False):
        self.name = name
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
                stepper = None # connect(stepper_ports[axis])
                log.info(f"Connected to {stepper_ports[axis]} as axis {axis}.")
            except ConnectionException as e:
                if require_connection:
                    raise e
                warnings.warn(f"Error opening stepper port {stepper_ports[axis]} as axis {axis}: {e}")
            
            self.axes[axis] = StageAxis(axis, piezo, stepper) 


        return

    def move(self, axis: str, movement: Position, which: Optional[MovementType] = None):
        result = self.axes[axis].move(movement, which)
        log.info(f"Stage {self.name}, Axis {axis} via move():" + result.text)
        return result


    def goto(self, axis: str, position: Position, which: Optional[MovementType] = None):
        result = self.axes[axis].goto(position, which)
        log.info(f"Stage {self.name}, Axis {axis} via goto():" + result.text)
        return result
