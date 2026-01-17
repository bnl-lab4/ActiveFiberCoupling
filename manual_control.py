"""
Input context for manual control of a stage.
"""

import time

from distance import Distance
from LoggingUtils import get_logger
from movement_classes import MovementType
from stage_status import run as status

# unique logger name for this module
logger = get_logger(__name__)


WHICH_DICT = dict(
    p=MovementType.PIEZO,
    piezo=MovementType.PIEZO,
    s=MovementType.STEPPER,
    stepper=MovementType.STEPPER,
    g=MovementType.GENERAL,
    general=MovementType.GENERAL,
)
AXES = ("x", "y", "z")
UNITS = ("microns", "volts", "steps", "fullsteps")
UNITS_ABRV = dict(
    u=UNITS[0],
    m=UNITS[0],
    v=UNITS[1],
    s=UNITS[2],
    st=UNITS[2],
    f=UNITS[3],
    fs=UNITS[3],
)


def run(stage, exposure_time):
    """
    Allows for manual control of a stage via text commands.

    Commands should consist of 3 or 4 space-separated arguments
    (case-insensitive): [axis] [device] [value] [unit, optional]. [axis]
    should be 'x', 'y', or 'z'. Acceptable values of [device] map to the
    values of `movement_classes.MovementType`, with first-letter
    abbreviations accepted. [value] is assumed to be in microns unless
    [unit] is specified as either 'microns', 'volts', 'steps', or
    'fullsteps', or first-letter abbreviations thereof. If in 'move>>'
    mode, [device] is move by [value]. If in 'goto>>' mode, [device] is
    moved to [value].  Besides an int or float, [value] can also be 'zero',
    'center', or 'max' to move the [device] to the lower limit, center, or
    upper limit of its travel, respectively. Change between 'move>>' and
    'goto>>' modes with keywords 'move' and 'goto'. To change the exposure
    time, enter 'texp [int]', where [int] is the desired exposure time in
    ~milliseconds (approximately, see `sensor_classes.Sensor.integrate`). An
    empty input string results in the integrated sensor value being
    printed. 'status' results in the stage status being printed. If in
    status mode ('status on', 'status off'), the position of the device
    that was moved will be printed after each move. 'q' exits the manual
    control context.

    Parameters
    ----------
    exposure_time : int, float
        Exposure time to use when integrating the sensor value.
    """

    print(
        """
'q' returns to menu, Enter key (empty input) returns integrated signal. 'texp' [int] changes exposure time.
Enter command as [axis] [device] [value], with an optional [unit] argument.
Arguments must be space separated. [device] can be 'piezo', 'stage', 'general, 'p', 's', or 'g'.
Units can be 'microns', 'volts', 'steps', 'fullsteps' (or 'u', 'v', 's', 'fs').
Piezo unit defaults to volts. Stepper and general units default to microns.
Value can be a number, 'zero', 'center', or 'max'. General uses stepper limits.
Examples: >>y piezo 15.2     >>x g -10     >>z p 5 v     >>x s 600 volts     >>y p center
Switch between goto and move (default) modes with 'goto' and 'move'.
"""
    )
    goto = False
    status_mode = False

    while True:
        input_msg = "goto >> " if goto else "move >> "
        user_input = input(input_msg).strip().lower()
        if user_input == "q":
            break
        if user_input == "":
            power = stage.sensor.integrate(exposure_time)
            power_str = str(f"{power:.6f}") if isinstance(power, float) else str(power)
            print(f"Power: {power_str}\nIntegrated for {exposure_time}")
            continue
        if user_input == "move":
            if goto:
                print("Switched to move mode.")
            else:
                print("Already in move mode.")
            goto = False
            continue
        if user_input == "goto":
            if goto:
                print("Already in goto mode.")
            else:
                print("Switched to goto mode.")
            goto = True
            continue

        if user_input == "status":
            status(stage, exposure_time, "all")
            continue
        if user_input == "status on":
            if status_mode:
                print("Status mode already on")
            else:
                print("Status mode turned on")
            status_mode = True
            continue
        if user_input == "status off":
            if status_mode:
                print("Status mode turned off")
            else:
                print("Status mode already off")
            status_mode = False
            continue

        user_input = user_input.split()
        if len(user_input) == 2 and set(user_input[0].lower()) == set("texp"):
            exposure_time = int(user_input[1])
            print(f'Exposure "time" set to {exposure_time}')
            continue
        if len(user_input) != 3 and len(user_input) != 4:
            print("Invalid input: Enter command as [axis] [device] [value] ([unit]).")
            continue

        if len(user_input) == 3:
            user_input.append("microns")

        axis, device, value, unit = user_input
        if axis.lower() not in AXES:
            print("Invalid input: Axis must be x or y or z")
            continue
        if device.lower() not in WHICH_DICT.keys():
            print("Invalid input: Device must be s, p, stepper, or piezo")
            continue

        if unit in UNITS:
            pass
        elif unit in UNITS_ABRV.keys():
            unit = UNITS_ABRV[unit]
        else:
            print("Invalid input: Unit must be u, v, s, fs")
            continue

        movement_type = WHICH_DICT[device]

        special = False
        if value.lower() == "zero":
            special = True
            if movement_type == MovementType.PIEZO:
                value = getattr(stage.axes[axis].piezo_limits[0], unit)
            else:
                value = getattr(stage.axes[axis].stepper_limits[0], unit)

        elif value.lower() == "max":
            special = True
            if movement_type == MovementType.PIEZO:
                value = getattr(stage.axes[axis].piezo_limits[1], unit)
            else:
                value = getattr(stage.axes[axis].stepper_limits[1], unit)

        elif value.lower() == "center":
            special = True
            if movement_type == MovementType.PIEZO:
                value = getattr(stage.axes[axis].piezo_center, unit)
            else:
                value = getattr(stage.axes[axis].stepper_center, unit)

        try:
            value = float(value)
        except Exception:
            print(
                "Invalid input: Value must be a real number or one of 'zero', 'center', 'max'"
            )
            continue

        if goto and value < 0:
            breakflag = False
            while True:
                print("Entered negative position while in goto mode, are you sure?")
                user_input = input("(y/n): ").strip().lower()
                if user_input == "y":
                    break
                if user_input == "n":
                    breakflag = True
                    break
                print("Could not interpret input")
            if breakflag:
                break

        if goto or special:
            _ = stage.goto(axis, Distance(value, unit), movement_type)
        else:
            _ = stage.move(axis, Distance(value, unit), movement_type)

        if status_mode:
            time.sleep(0.1)
            status(stage, exposure_time, movement_type.value, expose=False)
