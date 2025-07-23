from motion import move
from photodiode_in import getPower
from photodiode_in import get_exposure
import time
import numpy as np
import matplotlib.pyplot as plt
import csv

def scan(ser, step, file):
    xpos = 0
    zpos = 0
    count = 0
    #x_voltages = np.array([])
    x_voltages = []
    z_voltages = []
    #z_voltages = np.array([])
    powers = []
    move('x', xpos, ser)
    move('z', zpos, ser)
    while zpos <= 74:
        while xpos <= 74:
            move('x', xpos, ser)
            count += 1
            print("Count: ", count)
            #time.sleep(0.075)
            #power = getPower(1000)
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

    print(x_voltages)
    file.write(f"X Voltages: {x_voltages}\n")
    print(z_voltages)
    file.write(f"Z Voltages: {z_voltages}\n")
    print(powers)
    file.write(f"Power: {powers}\n")
    '''
    with open('data_run.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(x_voltages)
        writer.writerows(z_voltages)
        writer.writerows(powers)
    '''
    return x_voltages, z_voltages, powers

def intensity_plot(x_voltages, z_voltages, powers):
    x = np.array(x_voltages)
    z = np.array(z_voltages)
    power = np.array(powers)

    plt.scatter(x, z, s=power, alpha=0.5, edgecolors='blue')
    #plt.scatter(x, z)
    plt.title("Power intensity")
    plt.xlabel("X-axis")
    plt.ylabel("Z-axis")
    plt.show()

def find_max_intensity(x_voltages, z_voltages, powers, file):
    power_max = max(powers)
    power_max_index = powers.index(power_max)
    x_max = x_voltages[power_max_index]
    z_max = z_voltages[power_max_index]

    print(f"Max power: {power_max}")
    file.write(f"Max power: {power_max}\n")
    print(f"X position: {x_max}")
    file.write(f"X position: {x_max}\n")
    print(f"Z position: {z_max}")
    file.write(f"Z position: {z_max}\n")

    return(x_max, z_max, power_max)

def run(ser, file):
    print("Starting scan")
    step = 10

    y_values = [0, 37.5, 75]

    x_voltages1, z_voltages1, powers1 = scan(ser, step, file)
    #intensity_plot(x_voltages1, z_voltages1, powers1)
    
    #time.sleep(0.075)
    move('y', 37.5, ser)
    #time.sleep(1)
    x_voltages2, z_voltages2, powers2 = scan(ser, step, file)
    #intensity_plot(x_voltages2, z_voltages2, powers2)

    move('y', 75, ser)
    #time.sleep(1)
    x_voltages3, z_voltages3, powers3 = scan(ser, step, file)
    #intensity_plot(x_voltages3, z_voltages3, powers3)

    x_max1, z_max1, powers_max1 = find_max_intensity(x_voltages1, z_voltages1, powers1, file)
    x_max2, z_max2, powers_max2 = find_max_intensity(x_voltages2, z_voltages2, powers2, file)
    x_max3, z_max3, powers_max3 = find_max_intensity(x_voltages3, z_voltages3, powers3, file)

    all_powers = [powers_max1, powers_max2, powers_max3]
    all_x = [x_max1, x_max2, x_max3]
    all_z = [z_max1, z_max2, z_max3]

    # Initialize Variables
    '''Only fill in for the first and third plane during initialization'''

    x1, y1, z1, p1 = [x_max1, 0, z_max1, powers_max1] # First plane max point (y = 0.0)
    x2, y2, z2, p2 = [0, 37.5, 0, 0] # Second plane max point (y = 37.5)
    x3, y3, z3, p3 = [x_max3, 75, z_max3, powers_max3] # Third plane max point (y = 75.0)

    radius = 37.5
    error_margin = 1 # minimum radius size for search loop

    # Calculate slopes
    y2 = y1 + radius
    x_slope = (x3 - x1) / 75
    z_slope = (z3 - z1) / 75

    # Find center plane
    x = (x_slope * y1) + x1
    z = (z_slope * y1) + z1
    '''Find power value at (x, y2, z) and set as p2'''

    # Finding maximum power intensity
    y_values = [y1, y2, y3]
    power_values = [p1, p2, p3]
    max_power = max(power_values)
    max_index = power_values.index(max_power)

    # Set Center point values
    p2 = max_power # Set power for new center point (B)
    y2 = y_values[max_index] # set y-value for new center point (B)   

    # Search Loop
    while radius > error_margin:

        radius /= 2 # shrink radius

        # Finding power and y-value of left-bound point (A)
        if y2 != 0:
            y1 = y2 - radius
            x = (x_slope * y1) + x1
            z = (z_slope * y1) + z1
            '''Find power value at (x, y1, z) and set as p1'''
        else:
            y1 = 0
            p1 = 0
        
        # Finding power and y-value of right-bound point (C)
        if y2 != 75:
            y3 = y2 + radius
            x = (x_slope * y3) + x1
            z = (z_slope * y3) + z1
            '''Find power value at (x, y3, z) and set as p3'''
        else:
            y3 = 0
            p3 = 0

        # Finding maximum power intensity
        y_values = [y1, y2, y3]
        power_values = [p1, p2, p3]
        max_power = max(power_values)
        max_index = power_values.index(max_power)
    
        # Set Center point values
        p2 = max_power # Set power for new center point (B)
        y2 = y_values[max_index] # set y-value for new center point (B)   

    # Print final point
    x = (x_slope * y2) + x1
    z = (z_slope * y2) + z1
    Final_point = [x, y2, z, p2]

    print(f"Final x: {x}")
    print(f"Final z: {z}")
    print(f"Final y: {y2}")
    print(f"Final power: {p2}")

    
    '''
    powers_max = max(all_powers)
    powers_max_index = all_powers.index(powers_max)

    x_max = all_x[powers_max_index]
    z_max = all_z[powers_max_index]
    y_max = y_values[powers_max_index]
    
    print(f"Max power: {powers_max}")
    file.write(f"Max power: {powers_max}\n")
    print(f"X position: {x_max}")
    file.write(f"X position: {x_max}\n")
    print(f"Z position: {z_max}")
    file.write(f"Z position: {z_max}\n")
    print(f"Y position: {y_max}")
    file.write(f"Y position: {y_max}\n")
    
    move('x', x_max, ser)
    move('y', y_max, ser)
    move('z', z_max, ser)
    '''
    print:("Scan done")
