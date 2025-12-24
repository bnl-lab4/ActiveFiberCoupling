# TODO:
#   smart homing function
#   parallelize stepper movement
####################

"""
Defines the `StageAxis` and `StageDevices` classes for controlling stages.
"""

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
with open("stepper_info.yaml", "r") as stream:
    documents = yaml.load_all(stream, Loader=yaml.SafeLoader)
    stepper_info = next(documents)
    stepper_settings_check = next(documents)


class MovementType(enum.Enum):
    """
    An enumeration of the possible modes of movement.

    See `StageAxis` for more details.

    Members
    -------
    STEPPER : str
        Represents moving with only the stepper motors.
    PIEZO : str
        Represents moving only with the piezos.
    GENERAL : str
        Represents moving with both the steppers and piezos as needed.
    """

    STEPPER = "stepper"
    """Represents moving with only the stepper motors."""
    PIEZO = "piezo"
    """Represents moving with only the piezos."""
    GENERAL = "general"
    """Represents moving with both the steppers and piezos as needed."""


class MoveResult:
    """
    Enables logging of movements that may have been affected by several
    functions.

    This class carries information about movements from methods in child
    class `StageAxis` up to the parent class `StageDevices` for logging.

    Parameters
    ----------
    distance : `Distance.Distance`
        The distance by which the stepper or piezo was moved.
    movementType : `MovementType`
        Whether the movement was with the steppers or the piezos. The
        `movementType` cannot be ``GENERAL`` because that is not specific.
    centered_piezos : bool, default=False
        Whether the piezos were centered during the move. This will only be
        ``True`` if the initially requested movement type was ``GENERAL``
        and the final position was outside of the piezo's travel range.

    Attributes
    ----------
    text : str
        The text that should be in the log message regarding the movement.
    """

    def __init__(
        self,
        distance: Distance,
        movementType: MovementType,
        centered_piezos: bool = False,
    ):
        self.distance = distance
        self.movementType = movementType
        self.centered_piezos = centered_piezos

        # Any general move is composed of a piezo move and possibly
        #   a stepper move. It should be logged as such.
        assert self.movementType != MovementType.GENERAL, (
            "movementType cannot be general"
        )

    @property
    def text(self):
        """
        str : text for a log message regarding the movement.
        """
        values = []
        values.append(f"{self.distance.microns} microns")
        if self.movementType == MovementType.PIEZO:
            values.append(f"{self.distance.volts} volts")
        if self.movementType == MovementType.STEPPER:
            values.append(f"{self.distance.steps} steps")
            values.append(f"{self.distance.fullsteps} full steps")
        values = "(" + ", ".join(values) + ")"
        return (
            f"set {self.movementType.value} to {values}"
            + f"{'after centering piezos' if self.centered_piezos else ''}"
        )


class StageAxis:
    """
    Subclass for all movement devices on a single stage axis.

    Each axis of the NanoMax flexure stages from ThorLabs
    (<https://www.thorlabs.com/NewGroupPage9.cfm?ObjectGroup_ID=2386>)
    can be moved by a ThorLabs DRV208 stepper motor actuator
    (<https://www.thorlabs.com/thorproduct.cfm?partnumber=DRV208>)
    used with a Pololu T834 controller
    (<https://www.pololu.com/product/3132>) for coarse movement or by a
    piezo stack for fine movement. This class provides controls for the
    movement of a single stage axis, as a subclass of `StageDevices`.

    Parameters
    ----------
    axis : {'x', 'y', 'z'}
        Which axis this class instance will control.
    piezo : `serial.Serial`
        Serial object for the piezo controller.
    stepper : `ticlib.TicUSB`
        Serial object for the stepper controller.
    stepper_SN : str
        The serial number for the specific stepper control board.
    autohome : bool, default=True
        Whether to run the homing routine on upon establishing a
        connection to the stepper control board.

    Attributes
    ----------
    axis : {'x', 'y', 'z'}
        Which axis of the stage is controlled.
    piezo : `serial.Serial`
        Serial object for the piezo controller.
    stepper : `ticlib.TicUSB`
        Serial object for the stepper controller.
    stepper_SN : str
        The serial number for the specific stepper control board.
    autohome : bool, default=True
        Whether to run the homing routine on upon establishing a
        connection to the stepper control board.
    PIEZO_LIMITS : 2-tuple of `Distance.Distance`
        Upper and lower limits of the piezo travel range.
    PIEZO_CENTER : `Distance.Distance`
        Center of the piezo travel range.
    STEPPER_LIMITS : 2-tuple of `Distance.Distance`
        Upper and lower limits of the stepper travel range.
    STEPPER_CENTER : `Distance.Distance`
        Center of the stepper travel range.
    stepper_settings : dict
        Dictionary of the stepper control board settings.

    Methods
    -------
    energize()
        Energize the stepper motor.
    deenergize()
        Deenergize the stepper motor.
    home()
        Home the stepper motor (using the built-in limit switch).
    get_stepper_position()
        Get the current stepper position.
    get_piezo_position()
        Get the current piezo position.
    goto(position, which=None)
        Move the stepper and/or piezo to the desired position. ``Which``
        determines the movement type, which defaults to ``GENERAL``.
    move(movement, which=None)
        Move the stepper and/or piezo by the desired distance. ``Which``
        determines the movement type, which defaults to ``GENERAL``.
    """

    def __init__(self, axis: str, piezo, stepper, stepper_SN, autohome: bool = True):
        """
        Initializes `StageAxis` and checks stepper settings.

        Given the piezo and stepper serial objects and the stepper
        control board's serial number, the piezo and stepper travel range
        limits are defined and the stepper settings are compared to those
        in ``stepper_info.yaml``.

        Parameters
        ----------
        axis : {'x', 'y', 'z'}
            Which axis this class instance will control.
        piezo : `serial.Serial`
            Serial object for the piezo controller.
        stepper : `ticlib.TicUSB`
            Serial object for the stepper controller.
        stepper_SN : str
            The serial number for the specific stepper control board.
        autohome : bool, default=True
            Whether to run the homing routine on upon establishing a
            connection to the stepper control board.

        Notes
        -----
        While the specs of the 3-axis NanoMax flexure translation stage
        list the coarse movement travel range as 4 mm, several steppers
        have been observed to have longer ranges, some exceeding 5 mm.
        Additionally, the stepper motors travel range is 10mm, with the
        stage's 4 mm range roughly in the center of the stage's range,
        with some variation. For the steppers that have been measured, the
        lower range listed in the file is the number of steps from the
        limit switch until the actuator is engaged with stage, and the
        upper limit is the number of steps from the limit switch until the
        actuator pushes against the end of the stage's travel range. For
        steppers lacking this information in ``stepper_info.yaml``,
        conservative values are used. For ease of use, the stepper position
        is set so that zero is at the lower stage limit.
        """

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
                raise Exception(
                    f"Stepper {self.stepper_SN} step mode is set to "
                    + str(self.stepper.get_step_mode())
                    + " instead of the expected value of 5 (32 microsteps per step)."
                )
            self._step_mode_mult = 32  # currently going to enforce this

            #   getting stage limits from yaml file, if possible
            if stepper_SN in stepper_info.keys():
                self._TRUE_STEPPER_LIMITS = (
                    Distance(stepper_info[stepper_SN][0], "fullsteps"),
                    Distance(stepper_info[stepper_SN][1], "fullsteps"),
                )
            else:
                #   should be safe values
                self._TRUE_STEPPER_LIMITS = (
                    Distance(1000, "fullsteps"),
                    Distance(2800, "fullsteps"),
                )
                warnings.warn(
                    f"Stepper serial number {stepper} is not in stepper_info.yaml."
                    + "Stage range set to safe defaults."
                )
            log.debug(
                f"True stepper {self.stepper_SN} stage limits set to "
                + f"({self._TRUE_STEPPER_LIMITS[0].prettyprint()}, {self._TRUE_STEPPER_LIMITS[1].prettyprint()})"
            )

            self.STEPPER_LIMITS = (
                Distance(0, "fullsteps"),
                self._TRUE_STEPPER_LIMITS[1] - self._TRUE_STEPPER_LIMITS[0],
            )

            # stepper center is defined with lower limit = 0 steps
            self.STEPPER_CENTER = (self.STEPPER_LIMITS[1] - self.STEPPER_LIMITS[0]) / 2
            log.debug(
                f"Stepper {self.stepper_SN} stage center set to {self.STEPPER_CENTER.prettyprint()}"
            )

            #   checking that the stepper controller settings are what we want
            #       because we can't change most of them in python
            self.stepper_settings = {}
            self.stepper_settings.update(
                {
                    "rx limit switch reverse": self.stepper.settings.get_rx_limit_switch_reverse()
                }
            )
            self.stepper_settings.update(
                {"current limit": self.stepper.settings.get_current_limit()}
            )
            self.stepper_settings.update(
                {"max speed": self.stepper.settings.get_max_speed()}
            )
            self.stepper_settings.update(
                {"max acceleration": self.stepper.settings.get_max_acceleration()}
            )
            self.stepper_settings.update(
                {"max deceleration": self.stepper.settings.get_max_deceleration()}
            )
            self.stepper_settings.update(
                {"auto homing": self.stepper.settings.get_auto_homing()}
            )
            self.stepper_settings.update(
                {
                    "homing speed towards": self.stepper.settings.get_homing_speed_towards()
                }
            )
            self.stepper_settings.update(
                {"homing speed away": self.stepper.settings.get_homing_speed_away()}
            )
            self.stepper_settings.update(
                {"soft error response": self.stepper.settings.get_soft_error_response()}
            )
            self.stepper_settings.update(
                {"control mode": self.stepper.settings.get_control_mode()}
            )
            self.stepper_settings.update(
                {"command timeout": self.stepper.settings.get_serial_command_timeout()}
            )

            stepper_settings_fail = {}
            for key, value in self.stepper_settings.items():
                if value != stepper_settings_check[key]:
                    stepper_settings_fail.update(
                        {key: (value, stepper_settings_check[key])}
                    )
            if stepper_settings_fail != {}:
                warnings.warn(
                    f"Stepper {stepper_SN} settings not expected values:\n"
                    + "\n".join(
                        [
                            "setting  |  current  |  expected",
                        ]
                        + [
                            f"{key}  |  {value[0]}  |  {value[1]}"
                            for key, value in stepper_settings_fail.items()
                        ]
                    )
                )  # sorry
            else:
                log.debug(f"Stepper {stepper_SN} settings are as expected")

    def __enter__(self):
        """
        Energize and home (if enabled) the stepper upon entering context.

        Returns ``self`` for use in 'with... as...' statements.
        """
        if self.stepper is None:
            return self
        self.energize()
        if self.autohome:
            self.home()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Deenergize the stepper upon exiting context.

        Returns ``False``, letting exceptions propogate.
        """
        if self.stepper is None:
            return False
        self.deenergize()
        return False

    def _position_uncertain(self):
        """
        Get the ``position_uncertain`` flag from the stepper control board.

        If the motor is uncertain of its position (``True``) then move
        commands will be ignored. Successfully homing the stepper should
        result in ``position_uncertain`` being set to ``False``.

        Returns
        -------
        bool
            ``position_uncertain`` flag.
        """
        return (self.stepper.get_misc_flags()[0] >> 1) & 1

    def _energized(self):
        """
        Get the ``energized`` flag from the stepper control board.

        If the motor is not energized (``False``) then move commands
        will be ignored.

        Returns
        -------
        bool
            ``energized`` flag.
        """
        return (self.stepper.get_misc_flags()[0]) & 1

    def _homing_active(self):
        """
        Get the ``homing_active`` flag from the stepper control board.

        This flag is only true when the stepper is actively homing.

        Returns
        -------
        bool
            ``homing_active`` flag.
        """
        return (self.stepper.get_misc_flags()[0] >> 4) & 1

    def _goto_piezo(self, voltage: float) -> MoveResult:
        """
        Set the piezo to the desired position (voltage), if able.

        Parameters
        ----------
        voltage : float
            The voltage to set the piezo to. If this is outside the
            travel limits of the piezo, it will be set to the nearest
            limit.

        Returns
        -------
        `MoveResult`
            Info to be logged regarding the movement.
        """

        clamped = max(
            self.PIEZO_LIMITS[0].volts, min(self.PIEZO_LIMITS[1].volts, voltage)
        )  # piezo voltage limits
        if clamped != voltage:
            log.warning(
                f"Cannot move {self.axis} to {voltage} because it is"
                + " outside the piezo's limits of"
                + f"({self.PIEZO_LIMITS[0].volts}, {self.PIEZO_LIMITS[1].volts}) volts"
            )
        command = f"{self.axis.lower()}voltage={clamped}\n"
        self.piezo.read(self.piezo.in_waiting).decode("utf-8")
        self.piezo.write(command.encode())
        self.piezo.flush()

        return MoveResult(Distance(clamped, "volts"), MovementType.PIEZO)

    def _goto_stepper(self, steps: Union[int, float]) -> MoveResult:
        """
        Set the stepper to the desired position (steps).

        Parameters
        ----------
        steps : int, float
            The position in steps to move the motor to. If this is outside
            the travel limits of the stage, it will be moved to the
            nearest limit. If a float is provided, it is converted to an int.

        Returns
        -------
        `MoveResult`
            Info to be logged regarding the movement.
        """

        if not self._energized():
            raise RuntimeError(
                f"Axis {self.axis} stepper {self.stepper_SN} not energized"
            )

        if self._position_uncertain():
            raise RuntimeError(
                f"Axis {self.axis} stepper {self.stepper_SN} position is uncertain and needs homing"
            )

        clamped = max(
            self.STEPPER_LIMITS[0].steps, min(self.STEPPER_LIMITS[1].steps, steps)
        )  # stepper steps limits
        if clamped != steps:
            log.warning(
                f"Cannot move {self.axis} stepper {self.stepper_SN} to {steps} because it is"
                + "outside the stepper's stage limits of"
                + f"({self.STEPPER_LIMITS[0].steps}, {self.STEPPER_LIMITS[1].steps}) steps"
            )
        self.stepper.set_target_position(int(clamped))
        while self.stepper.get_target_position() != self.stepper.get_current_position():
            time.sleep(0.01)
        return MoveResult(Distance(steps, "steps"), MovementType.STEPPER)

    def energize(self):
        """
        Energize the stepper motor.

        Set the stepper motor to be in an energized state. The control
        board will ignore move commands if the stepper is not energized.
        """
        if self._energized():
            log.info(f"Axis {self.axis} stepper {self.stepper_SN} is already energized")
            return
        self.stepper.halt_and_set_position(0)
        self.stepper.energize()
        self.stepper.exit_safe_start()
        log.info(f"Axis {self.axis} stepper {self.stepper_SN} energized")

    def deenergize(self):
        """
        Deenergize the stepper motor.
        Set the motor to be in a deenergized state. The control board will
        ignore move commands until the board is re-energized.
        """
        if not self._energized():
            log.info(
                f"Axis {self.axis} stepper {self.stepper_SN} is already deenergized"
            )
            return
        self.stepper.halt_and_hold()
        self.stepper.deenergize()
        self.stepper.enter_safe_start()
        log.info(f"Axis {self.axis} stepper {self.stepper_SN} deenergized")

    def home(self):
        """
        Home the stepper.

        Run the homing routine for the stepper, then move the stepper to
        the lower stage limit and set that position to zero.

        Notes
        -----
        The homing routine uses a limit switch installed in the stepper
        to calibrate its position. First, it moves the motor towards the
        limit switch until it is active. Then, it slowly moves the motor
        away from the limit switch until it is no longer active. This
        position is then set to be the zero point.
        """
        if not self._energized():
            raise RuntimeError(
                f"Axis {self.axis} stepper {self.stepper_SN} not energized"
            )
        self.stepper.go_home(0)
        log.info(f"Axis {self.axis} stepper {self.stepper_SN} homing")
        while self._position_uncertain():
            # homing sequence sets "position uncertain" to false upon success
            time.sleep(0.1)

        #   set lower stage limit to 0
        self._goto_stepper(self._TRUE_STEPPER_LIMITS[0].steps)
        self.stepper.halt_and_set_position(0)
        log.debug(
            f"Axis {self.axis} stepper {self.stepper_SN} homing complete, "
            + f"zeroed at lower stage limit {self._TRUE_STEPPER_LIMITS[0].prettyprint()}"
        )

    def get_stepper_position(self):
        """
        Get the position of the stepper motor.

        Returns
        -------
        `Distance.Distance`
            Position of the stepper relative to the zero point (lower stage
            movement limit if stepper has been homed).
        """
        position = self.stepper.get_current_position()
        return Distance(position, "steps")

    def get_piezo_position(self):
        """
        Get the position of the piezo.

        Returns
        -------
        `Distance.Distance`
            Position of the piezo relative to the zero point.
        """
        self.piezo.read(8)  # seems to clear better than flushing?
        self.piezo.flush()
        self.piezo.flushInput()
        self.piezo.flushOutput()
        self.piezo.write(f"{self.axis}voltage?\n".encode())
        self.piezo.readline().decode("utf-8").strip()
        position = self.piezo.read(8).decode("utf-8").strip()[2:-1]
        return Distance(float(position), "volts")

    def goto(
        self, position: Distance, which: Optional[MovementType] = None
    ) -> MoveResult:
        """
        Move stepper and/or piezo to the desired position.

        If `which` is ``PIEZO``, the piezo will be set to `position`, taking
        ``PIEZO_LIMITS[0]`` to be zero. If `which` is ``STEPPER``, the
        stepper will be set to `position`, taking ``STEPPER_LIMITS[0]`` to
        be zero.

        If `which` is ``GENERAL``, then ``STEPPER_LIMITS[0]`` is
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
            return self._goto_piezo(position.volts)
        if which == MovementType.STEPPER:
            return self._goto_stepper(position.steps)

        if which == MovementType.GENERAL:
            stepper_position = self.get_stepper_position()

            # decide whether the position can be reached with only the piezos
            if (
                self.PIEZO_LIMITS[0]
                < position - stepper_position
                < self.PIEZO_LIMITS[1]
            ):
                return self._goto_piezo((position - stepper_position).volts)
            self._goto_piezo(self.PIEZO_CENTER.volts)
            stepper_target = position - self.PIEZO_CENTER
            result = self._goto_stepper(stepper_target.steps)
            result.centered_piezos = True
            return result

        raise ValueError("which must be a MovementType enum or None.")

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
            return self._goto_piezo((self.get_piezo_position() + movement).volts)
        if which == MovementType.STEPPER:
            return self._goto_stepper((self.get_stepper_position() + movement).steps)

        if which == MovementType.GENERAL:
            stepper_position = self.get_stepper_position()
            piezo_position = self.get_piezo_position()
            if self.PIEZO_LIMITS[0] < piezo_position + movement < self.PIEZO_LIMITS[1]:
                return self._goto_piezo((piezo_position + movement).volts)
            self._goto_piezo(self.PIEZO_CENTER.volts)
            stepper_target = (
                stepper_position + movement + (piezo_position - self.PIEZO_CENTER)
            )
            result = self._goto_stepper(stepper_target.steps)
            result.centered_piezos = True
            return result

        raise ValueError("which must be a MovementType enum or None.")


class StageDevices:
    """
    Parent class of all devices associated with a single stage.

    This class holds all of the subclasses for the devices associated with
    a single 3-axis NanoMax translation flexure stage from ThorLabs
    (<https://www.thorlabs.com/NewGroupPage9.cfm?ObjectGroup_ID=2386>).
    The associated devices for a complete stage are a sensor (see
    `SensorClasses.py`), a ThorLabs open-loop three-channel piezo
    controller
    (<https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=1191>),
    and three ThorLabs stepper motor actuators
    (<https://www.thorlabs.com/thorproduct.cfm?partnumber=DRV208>)
    controlled with three Pololu T834 controllers
    (<https://www.pololu.com/product/3132>).

    Parameters
    ----------
    name : str
        Name of the stage.
    piezo_port : str
        File path for the piezo controller connection.
    stepper_SNs : Dict[str, str]
        Dictionary of strings mapping axis labels (x, y, z) to their
        respective controller boards' serial numbers.
    sensor : `SensorClasses.Sensor`
        Sensor class to be used with the stage.
    piezo_baud_rate : int
        Baud rate of the piezo controller connection.
    require_connection : bool, default=False
        If ``True``, failure to connect to the piezo controller or the
        stepper controllers will raise an exception. Otherwise, these
        failures will be logged at the ``warning`` level.
    autohome : bool, default=True
        Whether to run the homing routine on upon establishing a
        connection to the stepper control board.

    Attributes
    ----------
    name : str
        Name of the stage.
    sensor : `SensorClasses.Sensor`
        Sensor used with the stage.
    axes : Dict[str, `StageAxis` or ``None``]
        Dictionary of stage axes, with the axis names as keys and
        `StageAxis` classes as values.
    piezo_port : str
        File path for the piezo controller connection.
    piezo_baud_rate : int
        Baud rate of the piezo controller connection.

    Methods
    -------
    energize(axes=None)
        Energize one or more stepper (default all).
    deenergize(axes=None)
        Deenergize one or more stepper (default all).
    home(axes=None):
        Home one or more stepper (default all).
    move(axis, movement, which=None)
        Move an axis by some amount using its stepper and/or piezo.
    goto(axis, position, which=None)
        Move an axis to a position using its stepper and/or piezo.
    read()
        Read the value of the sensor.
    integrate(Texp, avg=True)
        Read the sensor over `Texp` and return the sum or average (default).

    Notes
    -----
    For simulated stages, see `SimulationClasses.py`. This class is only
    for real, physical stages.
    """

    def __init__(
        self,
        name: str,
        piezo_port: str,
        stepper_SNs: Dict[str, str],
        sensor: Optional[Sensor] = None,
        piezo_baud_rate: int = 115200,
        require_connection: bool = False,
        autohome: bool = True,
    ):
        self.name = name
        self.sensor = sensor
        self.axes = {axis: None for axis in stepper_SNs.keys()}
        self.piezo_port = piezo_port
        self.piezo_baud_rate = piezo_baud_rate
        self._exit_stack = contextlib.ExitStack()  # for context management

        piezo = None
        try:
            piezo = serial.Serial(self.piezo_port, piezo_baud_rate, timeout=1)
            log.info(f"Connected to {piezo_port} at {piezo_baud_rate} baud.")
        except serial.SerialException as e:
            if require_connection:
                raise e
            log.warning(f"Could not open serial port {piezo_port}: {e}")

        # loop a similar try-except over the stepper controllers
        # while also creating the axis objects
        for axis, stepper_SN in stepper_SNs.items():
            try:
                if stepper_SN is not None:
                    stepper = TicUSB(product=0x00B5, serial_number=stepper_SNs[axis])
                    # Designation for Tic T834          Serial number (binary) of specific controller
                    log.info(f"Connected to {stepper_SNs[axis]} as axis {axis}.")
                else:
                    stepper = None
                    log.info(f"{self.name}:: no connection for {axis} provided")
            except Exception as e:
                if require_connection or str(e) != "USB device not found":
                    raise e
                log.warning(
                    f"Error opening stepper port {stepper_SNs[axis]} as axis {axis}: {e}"
                )
                stepper = None

            self.axes[axis] = StageAxis(axis, piezo, stepper, stepper_SN, autohome)

    def __enter__(self):
        """
        Enters the context management of each `StageAxis` in `axes`.

        Returns ``self`` for use in 'with... as...' statements.
        """
        for axis, stageAxis in self.axes.items():
            if stageAxis is not None:
                self._exit_stack.enter_context(stageAxis)
                log.debug(
                    f"Axis {axis} context entered for stepper {stageAxis.stepper_SN}"
                )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exits all open contexts.

        Returns ``False``, letting exceptions propogate.
        """
        self._exit_stack.close()
        log.debug("Exited context stack")
        return False

    def __str__(self):
        """Returns `name`."""
        return self.name

    def __repr__(self):
        """Returns `name`."""
        return self.name

    def deenergize(self, axes: Optional[str] = None):
        """
        Deenergize the steppers of one or more axes.

        Parameters
        ----------
        axes : str, optional
            Which axes' steppers to deenergize, all listed in a single
            string (e.g. 'xz'), or 'all' for all axes. ``None``
            defaults to 'all'.
        """
        if axes is None or axes.lower() == "all":
            axes = "xyz"
        axes = list(axes)
        for axis in axes:
            self.axes[axis].deenergize()

    def home(self, axes: Optional[str] = None):
        """
        Home the steppers of one or more axes.

        Parameters
        ----------
        axes : str, optional
            Which axes' steppers to home, all listed in a single
            string (e.g. 'xz'), or 'all' for all axes. ``None``
            defaults to 'all'.
        """
        if axes is None or axes.lower() == "all":
            axes = "xyz"
        axes = list(axes)
        for axis in axes:
            self.axes[axis].home()

    def energize(self, axes: Optional[str] = None):
        """
        Energize the steppers of one or more axes.

        Parameters
        ----------
        axes : str, optional
            Which axes' steppers to energize, all listed in a single
            string (e.g. 'xz'), or 'all' for all axes. ``None``
            defaults to 'all'.
        """
        if axes is None or axes.lower() == "all":
            axes = "xyz"
        axes = list(axes)
        for axis in axes:
            self.axes[axis].energize()

    def move(self, axis: str, movement: Distance, which: Optional[MovementType] = None):
        """
        Move `axis` by `movement` using its stepper and/or piezo.

        Parameters
        ----------
        axis : {'x', 'y', 'z'}
            Which axis to move.
        movement : `Distance.Distance`
            What distance to move the given axis.
        which : `MovementType`, optional
            Which movement type to use. See `StageAxis.move` for more
            information on the behaviour of the movement types.

        Returns
        -------
        `MovementResult`
            Info to be logged regarding the movement.
        """
        result = self.axes[axis].move(movement, which)
        log.trace(f"{self.name}, Axis {axis} :" + result.text)
        return result

    def goto(self, axis: str, position: Distance, which: Optional[MovementType] = None):
        """
        Move `axis` to `position` using its stepper and/or piezo.

        Parameters
        ----------
        axis : {'x', 'y', 'z'}
            Which axis to move.
        position : `Distance.Distance`
            What position to move the given axis to.
        which : `MovementType`, optional
            Which movement type to use. See `StageAxis.move` for more
            information on the behaviour of the movement types.

        Returns
        -------
        `MovementResult`
            Info to be logged regarding the movement.
        """
        result = self.axes[axis].goto(position, which)
        log.trace(f"{self.name} Axis {axis} : " + result.text)
        return result

    def read(self):
        """
        Read the value of the sensor.

        See `SensorClasses.Socket` and `SensorClasses.Piplate`.

        Returns
        -------
        float or int
            Value read from the sensor.
        """
        if self.sensor is None:
            log.warning("No sensor assigned to {self.name}")
            return None
        return self.sensor.read()

    def integrate(self, Texp: Union[int, float], avg: bool = True):
        """
        Integrate the value of the sensor of `Texp`.

        See `SensorClasses.Socket` and `SensorClasses.Piplate`.

        Returns
        -------
        float or int
            Integrate value from the sensor.
        """
        if self.sensor is None:
            log.warning("No sensor assigned to {self.name}")
            return None
        return self.sensor.integrate(Texp, avg)
