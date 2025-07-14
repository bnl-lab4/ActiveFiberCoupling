# Hill Climbing Algorithm
from motion import move
import time
import random
from photodiode_in import getPower

def climb(axis, ser, file):
    current_peak = random.uniform(0, 75)
    move_again(axis, current_peak, ser, file)
    current_value = getPower(1000)

    if axis == 'y':
        step = 3
    else:
        step = 1

    for i in range(1000):

        next_peak = current_peak + step
        move_again(axis, current_peak, ser, file)
        next_value = getPower(1000)

        if next_value > current_value:
            current_peak = next_peak
            current_value = next_value
        else:
            next_peak = current_peak - step
            move_again(axis, current_peak, ser, file)
            next_value = getPower(1000)
        if next_value > current_value:
            current_peak = next_peak
            current_value = next_value
        else:
            break
    move_again(axis, current_peak, ser, file)

    return current_peak

def check(axis, ser, current_peak, current_value, step, file):

    for i in range(1000):

        next_peak = current_peak + step
        move_again(axis, current_peak, ser, file)
        next_value = getPower(1000)

        if next_value > current_value:
            current_peak = next_peak
            current_value = next_value
        else:
            next_peak = current_peak - step
            move_again(axis, current_peak, ser, file)
            next_value = getPower(1000)
        if next_value > current_value:
            current_peak = next_peak
            current_value = next_value
        else:
            break

def move_again(axis, current_peak, ser, file):

    clamped = move(axis, current_peak, ser)
    file.write(f"Moved {axis.upper()} to {clamped:.3f}V\n")
    print(f"Moved {axis.upper()} to {clamped:.3f}V")
    file.write(f"Power: {getPower(1000)}\n")
    print(f"Power: {getPower(1000)}")

def fancy_climb(axis, ser, file):

    current_peak = random.uniform(0, 75)
    move_again(axis, current_peak, ser, file)
    current_value = getPower(1000)

    #if axis == 'y':
     #   step = 3
    #else:
     #   step = random.uniform(1, 10)
    step = random.uniform(1, 10)

    for i in range(10):      
        check(axis, ser, current_peak, current_value, step, file)
        last_peak = current_peak
        last_value = current_value

        current_peak = random.uniform(0, 75)
        move_again(axis, current_peak, ser, file)

        current_value = getPower(1000)

        step = random.uniform(1, 10)

        check(axis, ser, current_peak, current_value, step, file)
   
        if last_value > current_value:
            current_peak = last_peak
            current_value = last_value
        else:
            break

    move_again(axis, current_peak, ser, file)
  
    return current_peak

def run(ser, file):
    print("Running hill climbing algorithm")

    for axis in ['x', 'z', 'y']:
        #peak = climb(axis, ser, file)
        peak = fancy_climb(axis, ser, file)  
        #move_again(axis, current_peak, ser, file)
        time.sleep(0.05)

    print("Hill Climbing search complete.")
