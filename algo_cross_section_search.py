from motion import move
from photodiode_in import get_exposure
import time
import numpy as np
import matplotlib.pyplot as plt
import csv

# Main Function Code

def run(ser, file):

    # Initialize variables
    range_value = 75
    iter_value = 10
    max_x = 0 # x-coord of max power
    max_z = 0 # z-coord of max power
    powers_quad = [] # list of powers of quadrants
    count = 0 # Number of movements [OPTIONAL]
    subplot = 1 # Number of subplots iterated

    xpos = [] # x-coords of quadrants
    zpos = [] # z-coords of quadrants
    x_voltages = [] # Full list of x-coords searched [OPTIONAL]
    z_voltages = [] # Full list of z-coords searched [OPTIONAL]
    powers_list = [] # Full list of powers measured [OPTIONAL]

    while subplot > iter_value:

        # Calculate quadrant coordinates
        mid = range_value / 2
        step = range_value / 4
        xpos = [max_x, max_x + mid, max_x + mid, max_x]
        zpos = [max_z, max_z, max_z + mid, max_z + mid]
        
        # Search through quadrants
        for i in list(range(4)):
            move('x', xpos[i] + step, ser)
            move('z', zpos[i] + step, ser)
            x_voltages.append(xpos[i] + step) # [OPTIONAL]
            z_voltages.append(zpos[i] + step) # [OPTIONAL]
            p_value = get_exposure(100)
            powers_quad.append(p_value) 
            powers_list.append(p_value) # [OPTIONAL]
            # Print current point count [OPTIONAL]
            count += 1
            print("Count: ", count)
        
        # Find max power and respective coordinates
        max_power = max(powers_quad)
        max_index = powers_quad.index(max_power)
        max_x = xpos(max_index)
        max_z = zpos(max_index)
        powers_quad.clear()

        # Update range and increment subplot count
        range_value /= 2
        subplot += 1
    
    # Readjust max_x and max_z to true coordinates
    max_x += step
    max_z += step

    # Print and write x-, z-, and p- values [OPTIONAL]
    print(x_voltages)
    file.write(f"X Voltages: {x_voltages}\n")
    print(z_voltages)
    file.write(f"Z Voltages: {z_voltages}\n")
    print(powers_list)
    file.write(f"Power: {powers_list}\n")

    print(f"Max X: {max_x}")
    print(f"Max Z: {max_z}")
    print(f"Max power: {max_power}")

    #return max_x, max_z, max_power
