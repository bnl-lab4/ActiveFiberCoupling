import serial
from motion import move
import photodiode_in
from MovementClasses import MovementType, Position

def run(stage, expTime):
    print("""'q' returns to menu, 'ENTER' returns integrated SiPM (not yet).
Enter command as [axis] [value] [device], e.g. >>y 15.2 piezo or >>x -10 p.
Switch between goto (default) and move modes with 'goto' and 'move'.""")
    goto = True
    which_dict = dict(p=(MovementType.PIEZO, 'volts'), s=(MovementType.STEPPER, 'steps'),
                      piezo=(MovementType.PIEZO, 'volts'), stepper=(MovementType.STEPPER, 'steps'))
    axes = ('x', 'y', 'z')
    while True:
        user_input = input(">> ").strip()
        if user_input.lower() == 'q':
            break
        if user_input.lower() == 'move':
            if goto: print('Switched to move mode.')
            else: print('Already in move mode.')
            goto = False
            continue
        if user_input.lower() == 'goto':
            if goto: print('Already in goto mode.')
            else: print('Switched to goto mode.')
            goto = True
            continue

        user_input = user_input.split()
        if len(user_input) != 3:
            print('Invalid input: Enter command as [axis] [value] [device].')
            continue

        axis, value, device = user_input
        if axis.lower() not in axes:
            print('Invalid input: [axis] must be x or y or z')
            continue
        if device.lower() not in which_dict.keys():
            print('Invalid input: Device must be s, p, stepper, or piezo')
            continue
        
        if goto:
            stage.goto(axis, Position(value, which_dict[device][1]), which_dict[device][0])
        else:
            stage.move(axis, Position(value, which_dict[device][1]), which_dict[device][0])
