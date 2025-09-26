from MovementClasses import MovementType
from Distance import Distance


def run(stage, ExposureTime):
    print("""
'q' returns to menu, 'ENTER' returns integrated signal. 'texp' [int] changes exposure time.
Enter command as [axis] [device] [value], with an optional [unit] argument.
Arguments must be space separated. [device] can be 'piezo', 'stage', 'p', or 's'.
Units can be 'microns', 'volts', 'steps', 'fullsteps' (or 'u', 'v', 's', 'fs').
Piezo unit defaults to volts. Stepper unit defaults to full steps.
Examples: >>y piezo 15.2     >>x s -10     >>z p 5 v     >>x s 600 volts
Switch between goto (default) and move modes with 'goto' and 'move'.
""")
    goto = True
    WHICH_DICT = dict(p=(MovementType.PIEZO, 'volts'), s=(MovementType.STEPPER, 'steps'),
                      piezo=(MovementType.PIEZO, 'volts'), stepper=(MovementType.STEPPER, 'steps'))
    AXES = ('x', 'y', 'z')
    UNITS = ('microns', 'volts', 'steps', 'fullsteps')
    UNITS_ABRV = dict(u=UNITS[0], m=UNITS[0], v=UNITS[1],
                      s=UNITS[2], st=UNITS[2], f=UNITS[3], fs=UNITS[3])
    Texp = ExposureTime

    while True:
        user_input = input(">> ").strip().lower()
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
        else:
            unit = WHICH_DICT[device][1]

        if goto:
            stage.goto(axis, Distance(value, unit), WHICH_DICT[device][0])
        else:
            stage.move(axis, Distance(value, unit), WHICH_DICT[device][0])
