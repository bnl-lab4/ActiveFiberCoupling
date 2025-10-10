# TODO:
#   smart homing function
#   parallelize stepper movement
####################

import logging
import time
import serial
import warnings
import enum
import yaml
import contextlib
from ticlib import TicUSB
from typing import Dict, Optional, Union

from SensorClasses import Sensor
from Distance import Distance

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


class MoveResult:
    def __init__(self, distance: Distance, movementType: MovementType, centered_piezos: bool = False):
        self.distance = distance
        self.movementType = movementType
        self.centered_piezos = centered_piezos
        assert self.movementType != MovementType.GENERAL, "movementType cannot be general"

    @property
    def text(self):
        values = []
        values.append(f'{self.distance.microns} microns')
        if self.movementType == MovementType.PIEZO:
            values.append(f'{self.distance.volts} volts')
        if self.movementType == MovementType.STEPPER:
            values.append(f'{self.distance.steps} steps')
            values.append(f'{self.distance.fullsteps} full steps')
        values = '(' + ', '.join(values) + ')'
        return f"set {self.movementType.value} to {values}" + \
                    f"{'after centering piezos' if self.centered_piezos else ''}"


class StageAxis:

    def __init__(self, axis: str, piezo, stepper, stepper_SN, autohome: bool = True):
        self.axis = axis
        self.piezo = piezo
        self.stepper = stepper
        self.stepper_SN = stepper_SN
        self.autohome = autohome

        if self.piezo is not None:
            self.PIEZO_LIMITS = (Distance(0, "volts"), Distance(75, "volts"))
            self.PIEZO_CENTER = (self.PIEZO_LIMITS[1] - self.PIEZO_LIMITS[0]) / 2

        if self.stepper is not None:
            if self.stepper.get_step_mode() != 5:
                raise Exception(f"Stepper {self.stepper_SN} step mode is set to " +
                        str(self.stepper.get_step_mode()) +
                        " instead of the expected value of 5 (32 microsteps per step).")
            self.step_mode_mult = 32    # currently going to enforce this

            #   getting stage limits from yaml file, if possible
            if stepper_SN in stepper_info.keys():
                self.TRUE_STEPPER_LIMITS = (Distance(stepper_info[stepper_SN][0], 'fullsteps'),
                                       Distance(stepper_info[stepper_SN][1], 'fullsteps'))
            else:
                #   should be safe values
                self.TRUE_STEPPER_LIMITS = (Distance(1000, 'fullsteps'),
                                       Distance(2800, 'fullsteps'))
                warnings.warn(f"Stepper serial number {stepper} is not in stepper_info.yaml." +
                            "Stage range set to safe defaults.")
            log.debug(f"True stepper {self.stepper_SN} stage limits set to " +
            f"({self.TRUE_STEPPER_LIMITS[0].prettyprint()}, {self.TRUE_STEPPER_LIMITS[1].prettyprint()})")

            self.STEPPER_LIMITS = (Distance(0, 'fullsteps'),
                self.TRUE_STEPPER_LIMITS[1] - self.TRUE_STEPPER_LIMITS[0])

            # stepper center is defined with lower limit = 0 steps
            self.STEPPER_CENTER = (self.STEPPER_LIMITS[1] - self.STEPPER_LIMITS[0]) / 2
            log.debug(f"Stepper {self.stepper_SN} stage center set to {self.STEPPER_CENTER.prettyprint()}")

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
                warnings.warn(f"Stepper {stepper_SN} settings not expected values:\n" +
                        "\n".join(["setting  |  current  |  expected",] +
                        [f"{key}  |  {value[0]}  |  {value[1]}"
                        for key, value in stepper_settings_fail.items()]))  # sorry
            else:
                log.debug(f"Stepper {stepper_SN} settings are as expected")

    def __enter__(self):
        if self.stepper is None:
            return self
        self.energize()
        if self.autohome:
            self.home()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        #   gracefully stop and deenergize the stepper when done
        if self.stepper is None:
            return False
        self.deenergize()
        return False

    def _position_uncertain(self):
        return (self.stepper.get_misc_flags()[0] >> 1) & 1

    def _energized(self):
        return (self.stepper.get_misc_flags()[0]) & 1

    def _homing_active(self):
        return (self.stepper.get_misc_flags()[0] >> 4) & 1

    def _goto_piezo(self, voltage: float) -> MoveResult:
        clamped = max(self.PIEZO_LIMITS[0].volts,
                      min(self.PIEZO_LIMITS[1].volts, voltage)) #piezo voltage limits
        if clamped != voltage:
            log.warning(f"Cannot move {self.axis} to {voltage} because it is" +
            " outside the piezo's limits of" +
            f"({self.PIEZO_LIMITS[0].volts}, {self.PIEZO_LIMITS[1].volts}) volts")
        command = f"{self.axis.lower()}voltage={clamped}\n"
        self.piezo.read(self.piezo.in_waiting).decode("utf-8")
        self.piezo.write(command.encode())
        self.piezo.flush()

        return MoveResult(Distance(clamped, 'volts'), MovementType.PIEZO)

    def _goto_stepper(self, steps: int) -> MoveResult:
        if not self._energized():
            raise RuntimeError(f"Axis {self.axis} stepper {self.stepper_SN} not energized")

        if self._position_uncertain():
            raise RuntimeError(f"Axis {self.axis} stepper {self.stepper_SN} position is uncertain and needs homing")

        clamped = max(self.STEPPER_LIMITS[0].steps,
                      min(self.STEPPER_LIMITS[1].steps, steps)) #piezo steps limits
        if clamped != steps:
            log.warning(f"Cannot move {self.axis} stepper {self.stepper_SN} to {steps} because it is" +
            "outside the stepper's stage limits of" +
            f"({self.STEPPER_LIMITS[0].steps}, {self.STEPPER_LIMITS[1].steps}) steps")
        self.stepper.set_target_position(int(clamped))
        while self.stepper.get_target_position() != self.stepper.get_current_position():
            time.sleep(0.01)
        return MoveResult(Distance(steps, 'steps'), MovementType.STEPPER)

    def energize(self):
        if self._energized():
            log.info(f"Axis {self.axis} stepper {self.stepper_SN} is already energized")
            return
        self.stepper.halt_and_set_position(0)
        self.stepper.energize()
        self.stepper.exit_safe_start()
        log.info(f"Axis {self.axis} stepper {self.stepper_SN} energized")

    def deenergize(self):
        if not self._energized():
            log.info(f"Axis {self.axis} stepper {self.stepper_SN} is already deenergized")
            return
        self.stepper.halt_and_hold()
        self.stepper.deenergize()
        self.stepper.enter_safe_start()
        log.info(f"Axis {self.axis} stepper {self.stepper_SN} deenergized")

    def home(self):
        if not self._energized():
            raise RuntimeError(f"Axis {self.axis} stepper {self.stepper_SN} not energized")
        self.stepper.go_home(0)
        log.info(f"Axis {self.axis} stepper {self.stepper_SN} homing")
        while self._position_uncertain():
            # homing sequence sets "position uncertain" to false upon success
            time.sleep(0.1)

        #   set lower stage limit to 0
        self._goto_stepper(self.TRUE_STEPPER_LIMITS[0].steps)
        self.stepper.halt_and_set_position(0)
        log.debug(f"Axis {self.axis} stepper {self.stepper_SN} homing complete, " +
                    f"zeroed at lower stage limit {self.TRUE_STEPPER_LIMITS[0].prettyprint()}")

    def get_stepper_position(self):
        position = self.stepper.get_current_position()
        return Distance(position, "steps")

    def get_piezo_position(self):
        self.piezo.read(8)          #seems to clear better than flushing?
        self.piezo.flush()
        self.piezo.flushInput()
        self.piezo.flushOutput()
        self.piezo.write(f"{self.axis}voltage?\n".encode())
        self.piezo.readline().decode('utf-8').strip()
        position = self.piezo.read(8).decode('utf-8').strip()[2:-1]
        return Distance(float(position), 'volts')

    def goto(self, position: Distance, which: Optional[MovementType] = None) -> MoveResult:
        if which is None:
            which = MovementType.GENERAL

        if which == MovementType.PIEZO:
            return self._goto_piezo(position.volts)
        if which == MovementType.STEPPER:
            return self._goto_stepper(position.steps)

        if which == MovementType.GENERAL:
            stepper_position = self.get_stepper_position()

            # decide whether the position can be reached with only the piezos
            if self.PIEZO_LIMITS[0] < position - stepper_position < self.PIEZO_LIMITS[1]:
                return self._goto_piezo((position - stepper_position).volts)
            self._goto_piezo(self.PIEZO_CENTER.volts)
            stepper_target = position - self.PIEZO_CENTER
            result = self._goto_stepper(stepper_target.steps)
            result.centered_piezos = True
            return result

        raise ValueError("which must be a MovementType enum or None.")

    def move(self, movement: Distance, which: Optional[MovementType] = None) -> MoveResult:
        if which is None:
            which = MovementType.GENERAL

        if which == MovementType.PIEZO:
            return self._goto_piezo((self.get_piezo_position() + movement).volts)
        if which == MovementType.STEPPER:
            return self._goto_stepper((self.get_stepper_position() + movement).steps)

        if which == MovementType.GENERAL:
            stepper_position = self.get_stepper_position()
            piezo_position = self.get_piezo_position()
            if self.PIEZO_LIMITS[0] < piezo_position + movement < self.PIEZO_LIMITS[1]:
                return self._goto_piezo((piezo_position + movement).volts)
            self._goto_piezo(self.PIEZO_CENTER.volts)
            stepper_target = stepper_position + movement +\
                        (piezo_position - self.PIEZO_CENTER)
            result = self._goto_stepper(stepper_target.steps)
            result.centered_piezos = True
            return result

        raise ValueError("which must be a MovementType enum or None.")


class StageDevices:

    def __init__(self, name: str, piezo_port: str, stepper_SNs: Dict[str, str],
                 sensor: Sensor = None, piezo_baud_rate: int = 115200,
                 require_connection: bool = False, autohome: bool = True):
        self.name = name
        self.sensor = sensor
        self.axes = {axis : None for axis in stepper_SNs.keys()}
        self.piezo_port = piezo_port
        self.piezo_baud_rate = piezo_baud_rate
        self._exit_stack = contextlib.ExitStack()   # for context management

        piezo = None
        try:
            piezo = serial.Serial(self.piezo_port, piezo_baud_rate, timeout = 1)
            log.info(f"Connected to {piezo_port} at {piezo_baud_rate} baud.")
        except serial.SerialException as e:
            if require_connection:
                raise e
            log.warn(f"Could not open serial port {piezo_port}: {e}")

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
            except Exception as e:
                if require_connection or str(e) != 'USB device not found':
                    raise e
                log.warning(f"Error opening stepper port {stepper_SNs[axis]} as axis {axis}: {e}")
                stepper = None

            self.axes[axis] = StageAxis(axis, piezo, stepper, stepper_SN, autohome)

    def __enter__(self):
        for axis, stageAxis in self.axes.items():
            self._exit_stack.enter_context(stageAxis)
            log.debug(f"Axis {axis} context entered for stepper {stageAxis.stepper_SN}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._exit_stack.close()
        log.debug("Exited context stack")
        return False

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def deenergize(self, axes: Optional[str] = None):
        if axes is None or axes.lower() == 'all':
            axes = 'xyz'
        axes = list(axes)
        for axis in axes:
            self.axes[axis].deenergize()

    def home(self, axes: Optional[str] = None):
        if axes is None or axes.lower() == 'all':
            axes = 'xyz'
        axes = list(axes)
        for axis in axes:
            self.axes[axis].home()

    def energize(self, axes: Optional[str] = None):
        if axes is None or axes.lower() == 'all':
            axes = 'xyz'
        axes = list(axes)
        for axis in axes:
            self.axes[axis].energize()

    def move(self, axis: str, movement: Distance, which: Optional[MovementType] = None):
        result = self.axes[axis].move(movement, which)
        log.trace(f"{self.name}, Axis {axis} :" + result.text)
        return result

    def goto(self, axis: str, position: Distance, which: Optional[MovementType] = None):
        result = self.axes[axis].goto(position, which)
        log.trace(f"{self.name} Axis {axis} : " + result.text)
        return result

    def read(self):
        if self.sensor is None:
            log.warning("No sensor assigned to {self.name}")
            return None
        return self.sensor.read()

    def integrate(self, Texp: Union[int, float], avg: bool = True):
        if self.sensor is None:
            log.warning("No sensor assigned to {self.name}")
            return None
        return self.sensor.integrate(Texp, avg)
