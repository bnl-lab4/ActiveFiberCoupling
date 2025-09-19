#SOCKET TODO:
#   stepper integration
####################
import logging
import time
import serial
import warnings
import enum
import yaml
import contextlib
from ticlib import TicUSB
from typing import Dict, List, Optional, Union

from SensorClasses import Sensor, SensorType

# unique logger name for this module
log = logging.getLogger(__name__)

# load stepper info file
with open('stepper_info.yaml', 'r') as stream:
    documents = yaml.load_all(stream, Loader = yaml.SafeLoader)
    stepper_info = next(documents)
    stepper_settings_check = next(documents)


class MovementType(enum.Enum):
    STEPPER = "stepper"
    PIEZO = "piezo"
    GENERAL = "general"

class Distance:
    _MICRONS_PER_VOLT = 20 / 75
    _MICRONS_PER_FULL_STEP = 2.5     # I think
    #   microns per full step might be different for each stepper
    #   this would require some structural changes to handle
    _MICRONS_PER_STEP = _MICRONS_PER_FULL_STEP / 32 # enforcing this step mode in StageAxis __init__
    def __init__(self, value: Union[int, float], unit: str = "microns"):
        if unit == "microns":
            self._microns = float(value)
        elif unit == "volts":
            self._microns = float(value) * self._MICRONS_PER_VOLT
        elif unit == "steps":
            self._microns = float(value) * self._MICRONS_PER_STEP
        else:
            raise ValueError("Unsupported unit: unit must be 'microns', 'volts', or 'steps'")
        

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
    def __init__(self, position: Union[float, int], units: str, mechanism: str, centered_piezos: bool = False):
        self.position = position
        self.units = units
        self.mechanism = mechanism
        self.centered_piezos = centered_piezos


    @property
    def text(self):
        return f"set {self.mechanism} to {self.position} {self.units}" +\
                    f"{'after centering piezos' if self.centered_piezos else ''}"

class StageAxis:
    
    def __init__(self, axis: str, piezo, stepper, stepper_SN):
        self.axis = axis
        self.piezo = piezo
        self.stepper = stepper
        self.stepper_SN = stepper_SN

        if self.stepper is not None:
            if self.stepper.get_step_mode() != 5:
                raise Exception(f"Stepper {self.stepper_SN} step mode is set to " +\
                        str(self.stepper.get_step_mode()) +\
                        " instead of the expected value of 5 (32 microsteps per step).")
            self.step_mode_mult = 32    # currently going to enforce this

            #   getting stage limits from yaml file, if possible
            self.PIEZO_LIMITS = (Distance(0, "volts"), Distance(75, "volts"))
            self.PIEZO_CENTER = (self.PIEZO_LIMITS[1] - self.PIEZO_LIMITS[0]) / 2

            if stepper_SN in stepper_info.keys():
                self.STEPPER_LIMITS = (Distance(stepper_info[stepper_SN][0] * self.step_mode_mult, 'steps'),
                                        Distance(stepper_info[stepper_SN][1] * self.step_mode_mult, 'steps'))
            else:
                #   should be safe values
                self.STEPPER_LIMITS = (Distance(1000 * self.step_mode_mult, 'steps'),
                                        Distance(2800 * self.step_mode_mult, 'steps'))
                warnings.warn(f"Stepper serial number {stepper} is not in stepper_info.yaml." +\
                            "Stage range set to safe defaults.")
            log.debug(f"Stepper {self.stepper_SN} stage limits set to ({self.STEPPER_LIMITS[0].steps}, {self.STEPPER_LIMITS[1].steps})")
            self.STEPPER_CENTER = (self.STEPPER_LIMITS[1] - self.STEPPER_LIMITS[0]) / 2 + self.STEPPER_LIMITS[0]
            log.debug(f"Stepper {self.stepper_SN} stage center set to {self.STEPPER_CENTER.steps}")

            #   checking that the stepper controller settings are what we want
            #       because we can't change most of them in python
            self.stepper_settings = {}
            self.stepper_settings.update(
                    {"rx limit switch reverse" : self.stepper.settings.get_rx_limit_switch_reverse()})
            self.stepper_settings.update(
                    {"current limit" : self.stepper.settings.get_current_limit()})
            self.stepper_settings.update(
                    {"max speed" : self.stepper.settings.get_max_speed()})
            self.stepper_settings.update(
                    {"max acceleration" : self.stepper.settings.get_max_acceleration()})
            self.stepper_settings.update(
                    {"max deceleration" : self.stepper.settings.get_max_deceleration()})
            self.stepper_settings.update(
                    {"auto homing" : self.stepper.settings.get_auto_homing()})
            self.stepper_settings.update(
                    {"homing speed towards" : self.stepper.settings.get_homing_speed_towards()})
            self.stepper_settings.update(
                    {"homing speed away" : self.stepper.settings.get_homing_speed_away()})
            self.stepper_settings.update(
                    {"soft error response" : self.stepper.settings.get_soft_error_response()})
            self.stepper_settings.update(
                    {"control mode" : self.stepper.settings.get_control_mode()})
            self.stepper_settings.update(
                    {"command timeout" : self.stepper.settings.get_serial_command_timeout()})

            stepper_settings_fail = {}
            for key, value in self.stepper_settings.items():
                if value != stepper_settings_check[key]:
                    stepper_settings_fail.update({key : (value, stepper_settings_check[key])})
            if stepper_settings_fail != {}:
                warnings.warn(f"Stepper {stepper_SN} settings not expected values:\n" +\
                        "\n".join(["setting  |  current  |  expected",] +\
                        [f"{key}  |  {value[0]}  |  {value[1]}"\
                        for key, value in stepper_settings_fail.items()]))  # sorry
            else:
                log.info(f"Stepper {stepper_SN} settings are as expected")


    def __enter__(self):
        if self.stepper is None:
            return self
        self.stepper.halt_and_set_position(0)
        self.stepper.energize()
        self.stepper.exit_safe_start()
        self.stepper.go_home(0)
        log.info(f"Stepper {self.stepper_SN} energized and now homing")
        while (self.stepper.get_misc_flags()[0] >> 1) & 1:
            # check whether "position uncertain" flag is true
            # homing sequence sets "position uncertain" to false upon success
            time.sleep(0.1)
        log.info(f"Stepper {self.stepper_SN} homing complete")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        #   gracefully stop and deenergize the stepper when done
        if self.stepper is None:
            return False
        self.stepper.halt_and_hold()
        self.stepper.deenergize()
        self.stepper.enter_safe_start()
        log.info(f"Deenergized stepper {self.stepper_SN} upon exit")
        return False

    def _goto_piezo(self, voltage: float) -> float:
        clamped = max(0.0, min(75.0, voltage)) #piezo voltage limits
        if clamped != voltage:
            warnings.warn(f"Requested {self.axis.upper()}={voltage:.2f}V, clamped to {clamped:.2f}V")
        command = f"{self.axis.lower()}voltage={clamped}\n"
        self.piezo.read(self.piezo.in_waiting).decode("utf-8")
        self.piezo.write(command.encode())
        self.piezo.flush()

        return MoveResult(clamped, 'volts', 'piezo')

    def _goto_stepper(self, steps: int) -> float:
        # send move command to stepper
        # include clamping to safe values (per stepper)
        # this is a goto function
        steps = int(steps)
        if steps > self.STEPPER_LIMITS[1].steps:
            warnings.warn(f"Cannot move {self.stepper_SN} to {steps} because it is above the steppers "+\
                    f"stage limit of {self.STEPPER_LIMITS[1].steps}")
        else:
            self.stepper.set_target_position(steps)
            while self.stepper.get_target_position() != self.stepper.get_current_position():
                time.sleep(0.01)
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
               return self._goto_piezo(position.volts)
            self._goto_piezo(self.PIEZO_CENTER.volts)
            result = self._goto_stepper((position - (piezo_position + self.self.PIEZO_CENTER)).steps)
            result.centered_piezos = True
            return result

        if which == MovementType.PIEZO:
            return self._goto_piezo(position.volts)
        if which == MovementType.STEPPER:
            return self._goto_stepper(position.steps)

        raise ValueError("which must be a MovementType enum or None.")

    def move(self, movement: Distance, which: Optional[MovementType] = None) -> float:
        # get stepper and piezo positions as Distance objects
        stepper_position = Distance(self.stepper.get_current_position(), 'steps')

        self.piezo.read(8)          #seems to clear better than flushing?
        self.piezo.flush()
        self.piezo.flushInput()
        self.piezo.flushOutput()
        self.piezo.write(f"{self.axis}voltage?\n".encode())
        self.piezo.readline().decode('utf-8').strip()
        piezo_position = self.piezo.read(8).decode('utf-8').strip()[2:-1] 
        piezo_position = Distance(float(piezo_position), 'volts')

        if which == MovementType.GENERAL or which is None:
            axis_position = stepper_position + piezo_position

            if 0 < (axis_position + movement).volts < 75:
                return self._goto_piezo((piezo_position + movement).volts)
            self._goto_piezo(self.PIEZO_CENTER.volts)
            result = self._goto_stepper((stepper_position + movement -\
                    (piezo_position - self.self.PIEZO_CENTER)).steps)
            result.centered_piezos = True
            return result

        if which == MovementType.PIEZO:
            return self._goto_piezo((piezo_position + movement).volts)
        if which == MovementType.STEPPER:
            return self._goto_stepper((stepper_position + movement).steps)

        raise ValueError("which must be a MovementType enum or None.")
            
            

class StageDevices:
    
    def __init__(self, name: str, piezo_port: str, stepper_SNs: Dict[str, str], sensor: Sensor = None,
                 piezo_baud_rate: int = 115200, require_connection: bool = False):
        self.name = name
        self.sensor = sensor
        self.axes = {axis : None for axis in stepper_SNs.keys()}
        self.piezo_port = piezo_port
        self._exit_stack = contextlib.ExitStack()   # for context management


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
        for axis, stepper_SN in stepper_SNs.items():
            try:
                if stepper_SN is not None:
                    stepper = TicUSB(product = 0x00b5, serial_number = stepper_SNs[axis])
                    # Designation for Tic T834          Serial number (binary) of specific controller
                    log.info(f"Connected to {stepper_SNs[axis]} as axis {axis}.")
                else:
                    stepper = None
                    log.info(f"{self.name}:: no connection for {axis} provided")
            except ConnectionException as e:
                if require_connection:
                    raise e
                warnings.warn(f"Error opening stepper port {stepper_SNs[axis]} as axis {axis}: {e}")
            
            self.axes[axis] = StageAxis(axis, piezo, stepper, stepper_SN) 



    def __enter__(self):
        for axis, stageAxis in self.axes.items():
            self._exit_stack.enter_context(stageAxis)
            log.debug(f"Axis {axis} context entered for stepper {stageAxis.stepper_SN}")
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        self._exit_stack.close()
        log.debug("Exited context stack gracefully")
        return False


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

