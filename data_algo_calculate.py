from motion import move
from photodiode_in import getPower
from photodiode_in import get_exposure
import time
import numpy as np
import matplotlib.pyplot as plt
import csv

def scan(ser, step, file, plane = 0):
    
    xpos = 0
    zpos = 0
    count = 0

    x_voltages = []
    z_voltages = []
    
    powers = []
    move('x', xpos, ser)
    move('z', zpos, ser)
    
    while zpos <= 74:
        while xpos <= 74:
            
            move('x', xpos, ser)
            count += 1
            print("Count: ", count)
            #time.sleep(0.075)
            
            power = get_exposure(100)
            x_voltages.append(xpos)
            z_voltages.append(zpos)
            powers.append(power)
            xpos += step
        
        xpos = 0
        zpos += step
        move('z', zpos, ser)
        count += 1
        print("Count: ", count)
        #time.sleep(0.075)

    print(f"\nPlane {plane}:")
    file.write(f"\nPlane {plane}:\n")
    print(f"X Voltages: {x_voltages}")
    file.write(f"X Voltages: {x_voltages}\n")
    print(f"Z Voltages: {z_voltages}")
    file.write(f"Z Voltages: {z_voltages}\n")
    print(f"Power: {powers}")
    file.write(f"Power: {powers}\n")

    return x_voltages, z_voltages, powers

def intensity_plot(x_voltages, z_voltages, powers):
    
    x = np.array(x_voltages)
    z = np.array(z_voltages)
    power = np.array(powers)

    plt.scatter(x, z, s=power, alpha=0.5, edgecolors='blue')
    plt.title("Power intensity")
    plt.xlabel("X-axis")
    plt.ylabel("Z-axis")
    plt.show()

def find_max_intensity(x_voltages, z_voltages, powers, file, plane = 0):
    
    power_max = max(powers)
    index = powers.index(power_max)
    x_max = x_voltages[index]
    z_max = z_voltages[index]

    print(f"\nPlane {plane} max point (x, z, power): ({x_max}, {z_max}, {power_max})")
    file.write(f"\nPlane {plane} Max point (x, z, power): ({x_max}, {z_max}, {power_max})\n")

    return(x_max, z_max, power_max)

def run(ser, file):

    # Scan planes 1 and 3
    print("Starting scan")
    step = 5

    x_voltages1, z_voltages1, powers1 = scan(ser, step, file, 1)
    #intensity_plot(x_voltages1, z_voltages1, powers1, 1)

    move('y', 75, ser)
    #time.sleep(1)
    
    x_voltages3, z_voltages3, powers3 = scan(ser, step, file, 3)
    #intensity_plot(x_voltages3, z_voltages3, powers3, 3)

    x_max1, z_max1, powers_max1 = find_max_intensity(x_voltages1, z_voltages1, powers1, file)
    x_max3, z_max3, powers_max3 = find_max_intensity(x_voltages3, z_voltages3, powers3, file)

    # Initialize Variables
    x1, y1, z1, p1 = [x_max1, 0, z_max1, powers_max1] # First plane max point (y = 0.0)
    x3, y3, z3, p3 = [x_max3, 75, z_max3, powers_max3] # Third plane max point (y = 75.0)

    radius = 37.5
    error_margin = 1 # minimum radius size for search loop

    # Calculate slopes
    x_slope = (x3 - x1) / 75
    z_slope = (z3 - z1) / 75

    # Find plane 2
    y2 = y1 + radius
    x = (x_slope * y2) + x1
    z = (z_slope * y2) + z1
    
    move('x', x, ser)
    move('y', y2, ser)
    move('z', z, ser)
    p2 = get_exposure(100)

    # Finding maximum power intensity
    y_values = [y1, y2, y3]
    power_values = [p1, p2, p3]
    max_power = max(power_values)
    max_index = power_values.index(max_power)

    # Set Center point values
    p2 = max_power # Set power for new center point (B)
    y2 = y_values[max_index] # set y-value for new center point (B)   

    # Search Loop
    cross_power = []
    cross_y = []

    while radius > error_margin:

        radius /= 2 # shrink radius

        # Finding power and y-value of left-bound point (A)
        if y2 != 0:
            y1 = y2 - radius
            x = (x_slope * y1) + x1
            z = (z_slope * y1) + z1
            move('x', x, ser)
            move('y', y1, ser)
            move('z', z, ser)
            p1 = get_exposure(100)
        else: y1 = 0; p1 = 0
        
        # Finding power and y-value of right-bound point (C)
        if y2 != 75:
            y3 = y2 + radius
            x = (x_slope * y3) + x1
            z = (z_slope * y3) + z1
            move('x', x, ser)
            move('y', y3, ser)
            move('z', z, ser)
            p3 = get_exposure(100)
        else: y3 = 0; p3 = 0

        # Finding maximum power intensity
        y_values = [y1, y2, y3]
        power_values = [p1, p2, p3]
        max_power = max(power_values)
        max_index = power_values.index(max_power)
    
        # Set Center point values
        p2 = max_power # Set power for new center point (B)
        y2 = y_values[max_index] # set y-value for new center point (B)
        # Record Values
        cross_power.append(p2)
        cross_y.append(y2)

    # Print final point
    x = (x_slope * y2) + x1
    z = (z_slope * y2) + z1

    move('x', x, ser)
    move('y', y2, ser)
    move('z', z, ser)

    print("\nCross-Plane Search:")
    file.write("\nCross-Plane Search:\n")
    print(f"Y Voltages: {cross_y}")
    file.write(f"Y Voltages: {cross_y}\n")
    print(f"Power: {cross_power}")
    file.write(f"Power: {cross_power}\n")
    
    print(f"\nFinal Max Point (x, y, z, power): ({x}, {y2}, {z}, {p2}")
    file.write(f"\nFinal Max Point (x, y, z, power): ({x}, {y2}, {z}, {p2}\n")
    print("Scan done")