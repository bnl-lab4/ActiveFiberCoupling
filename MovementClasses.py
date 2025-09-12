########### TODO:
# each axis gets a class, then each stage gets a parent class
####################
import logging
import time
import serial
import warnings
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
        if unit == "microns":
            self._microns = float(value)
        elif unit == "volts":
            self._microns = float(value) * MICRONS_PER_VOLT
        elif unit == "steps":
            self._microns = float(value) * MICRONS_PER_STEP
        else:
            raise ValueError("Unsupported unit: unit must be 'microns', 'volts', or 'steps'")
        
        return

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


class StageAxis:
    
    CENTER = Position(37.5, "volts")

    def __init__(self, axis: str, piezo, stepper):
        self.axis = axis
        self.piezo = piezo
        self.stepper = stepper
        return

    def _move_piezo(self, voltage: float) -> float:
        clamped = max(0.0, min(75.0, voltage)) #piezo voltage limits
        if clamped != position:
            warnings.warn(f"Requested {axis.upper()}={voltage:.2f}V, clamped to {clamped:.2f}V")
        command = f"{self.axis.lower()}voltage={clamped}\n"
        piezo.read(piezo.in_waiting).decode("utf-8")
        piezo.write(command.encode())
        piezo.flush()

        return MoveResult(clamped, 'volts', 'piezo')

    def _move_stepper(self, steps: float) -> float:
        # send move command to stepper
        # clamping might be good if we can identify the bounds of the stepper
        # I'm assuming this is a goto function and not a moveby function
        return MoveResult(steps, 'steps', 'stepper')

    def goto(self, position: Position, which: Optional[MovementType] = None) -> float:
        # get stepper and piezo positions as Position objects
        stepper_position = Position(1, 'steps')
        piezo_position = Position (0, 'volts')

        if which == MovementType.GENERAL or is None:
            # decide whether the position can be reached with only the piezos
            if 0 < stepper_position.volts - movement.volts < 75:
               return _move_piezo(position.volts)
           _move_piezo(CENTER.volts)
           result = _move_stepper(position.steps - (piezo_position.steps + CENTER.steps))
           result.centered_piezos = True
           return result

       if which == MovementType.PIEZO:
           return _move_piezo(position.volts), centered_piezos
       if which == MovementType.STEPPER:
           return _move_stepper(position.steps), centered_piezos

    def move(self, movement: Position, which: Optional[MovementType] = None) -> float:
        # get stepper and piezo positions as Position objects
        stepper_position = Position(1, 'steps')
        piezo_position = Position (0, 'volts')
        axis_position = Position(stepper_position.microns + piezo_position.microns, 'microns')

        if which == MovementType.GENERAL or is None:
            if 0 < movement.volts < 75:
                return _move_piezo(piezo_position.volts + movement.volts), centered_piezos
            _move_piezo(CENTER.volts)
            result = _move_stepper(stepper_position.steps + movement.steps -\
                    (piezo_position.steps - CENTER.steps)), centered_piezos
            result.centered_piezos = True
            return result

        if which == MovementType.PIEZO:
            return _move_piezo(piezo_position.volts + movement.volts), centered_piezos
        if which == MovementType.STEPPER:
            return _move_stepper(stepper_position.steps + movement.steps), centered_piezos
            
            

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
                # stepper = connect(stepper_ports[axis])
                log.info(f"Connected to {stepper_ports[axis]} as axis {axis}.")
            except ConnectionException as e:
                if require_connection:
                    raise e
                warnings.warn(f"Error opening stepper port {stepper_ports[axis]} as axis {axis}: {e}")
            
            self.axes[axis] = StageAxis(axis, piezo, stepper) 


        return

    def move(axis: str, movement: Position, which: Optional[MovementType] = None):
        result = self.axes[axis].move(movement, which)
        log.info(f"Stage {self.name}, Axis {axis}:" +\
                f"set {result.mechanism} to {result.position} {result.units} via move()" +\
                f"{'after centering piezos' if centered_piezos else ''}")


    def goto(axis: str, position: Position, which: Optional[MovementType] = None):
        result = self.axes[axis].move(position, which)
        log.info(f"Stage {self.name}, Axis {axis}:" +\
                f"set {result.mechanism} to {result.position} {result.units} via goto()" +\
                f"{'after centering piezos' if centered_piezos else ''}")
