import time

from StageStatus import run as status
from MovementClasses import MovementType
from Distance import Distance


def run(stage, ExposureTime):
    print("""
'q' returns to menu, 'ENTER' returns integrated signal. 'texp' [int] changes exposure time.
Enter command as [axis] [device] [value], with an optional [unit] argument.
Arguments must be space separated. [device] can be 'piezo', 'stage', 'general, 'p', 's', or 'g'.
Units can be 'microns', 'volts', 'steps', 'fullsteps' (or 'u', 'v', 's', 'fs').
Piezo unit defaults to volts. Stepper and general units default to microns.
Value can be a number, 'zero', 'center', or 'max'. General uses stepper limits.
Examples: >>y piezo 15.2     >>x g -10     >>z p 5 v     >>x s 600 volts     >>y p center
Switch between goto and move (default) modes with 'goto' and 'move'.
""")
    goto = False
    status_mode = False
    WHICH_DICT = dict(p=(MovementType.PIEZO, 'volts'), piezo=(MovementType.PIEZO, 'volts'),
                      s=(MovementType.STEPPER, 'microns'), stepper=(MovementType.STEPPER, 'microns'),
                      g=(MovementType.GENERAL, 'microns'), general=(MovementType.GENERAL, 'microns'))
    # string in tuples in WHICH_DICT are default units for that devic
    AXES = ('x', 'y', 'z')
    UNITS = ('microns', 'volts', 'steps', 'fullsteps')
    UNITS_ABRV = dict(u=UNITS[0], m=UNITS[0], v=UNITS[1],
                      s=UNITS[2], st=UNITS[2], f=UNITS[3], fs=UNITS[3])
    Texp = ExposureTime

    while True:
        input_msg = 'goto >> ' if goto else 'move >> '
        user_input = input(input_msg).strip().lower()
        if user_input == 'q':
            break
        if user_input == '':
            power = stage.sensor.integrate(Texp)
            power_str = str(f"{power:.6f}") if isinstance(power, float) else str(power)
            print(f"Power: {power_str}\nIntegrated for {Texp}")
            continue
        if user_input == 'move':
            if goto:
                print('Switched to move mode.')
            else:
                print('Already in move mode.')
            goto = False
            continue
        if user_input == 'goto':
            if goto:
                print('Already in goto mode.')
            else:
                print('Switched to goto mode.')
            goto = True
            continue

        if user_input == 'status':
            status(stage, Texp, 'all')
            continue
        if user_input == 'status on':
            if status_mode:
                print('Status mode already on')
            else:
                print('Status mode turned on')
            status_mode = True
            continue
        if user_input == 'status off':
            if status_mode:
                print('Status mode turned off')
            else:
                print('Status mode already off')
            status_mode = False
            continue

        user_input = user_input.split()
        if len(user_input) == 2 and set(user_input[0].lower()) == set('texp'):
            Texp = int(user_input[1])
            print(f'Exposure "time" set to {Texp}')
            continue
        if len(user_input) != 3 and len(user_input) != 4:
            print('Invalid input: Enter command as [axis] [device] [value] ([unit]).')
            continue

        if len(user_input) == 3:
            user_input.append(None)

        axis, device, value, unit = user_input
        if axis.lower() not in AXES:
            print('Invalid input: Axis must be x or y or z')
            continue
        if device.lower() not in WHICH_DICT.keys():
            print('Invalid input: Device must be s, p, stepper, or piezo')
            continue

        if unit is not None:
            if unit in UNITS:
                pass
            elif unit in UNITS_ABRV.keys():
                unit = UNITS_ABRV[unit]
            else:
                print('Invalid input: Unit must be u, v, s, fs')
                continue
        else:
            unit = WHICH_DICT[device][1]

        movetype = WHICH_DICT[device][0]

        special = False
        if value.lower() == 'zero':
            special = True
            if movetype == MovementType.PIEZO:
                value = getattr(stage.axes[axis].PIEZO_LIMITS[0], unit)
            else:
                value = getattr(stage.axes[axis].STEPPER_LIMITS[0], unit)

        elif value.lower() == 'max':
            special = True
            if movetype == MovementType.PIEZO:
                value = getattr(stage.axes[axis].PIEZO_LIMITS[1], unit)
            else:
                value = getattr(stage.axes[axis].STEPPER_LIMITS[1], unit)

        elif value.lower() == 'center':
            special = True
            if movetype == MovementType.PIEZO:
                value = getattr(stage.axes[axis].PIEZO_CENTER, unit)
            else:
                value = getattr(stage.axes[axis].STEPPER_CENTER, unit)

        try:
            value = float(value)
        except Exception:
            print('Invalid input: Value must be a real number')
            continue

        if goto and value < 0:
            breakflag = False
            while True:
                print("Entered negative position while in goto mode, are you sure?")
                user_input = input('(y/n): ').strip().lower()
                if user_input == 'y':
                    break
                if user_input == 'n':
                    breakflag = True
                    break
                print("Could not interpret input")
            if breakflag:
                break

        if goto or special:
            _ = stage.goto(axis, Distance(value, unit), movetype)
        else:
            _ = stage.move(axis, Distance(value, unit), movetype)

        if status_mode:
            time.sleep(0.1)
            status(stage, Texp, movetype.value, expose=False)
