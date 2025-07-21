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
            x_voltages.append(xpos)
            z_voltages.append(zpos)
            powers.append(get_exposure(100))
            xpos += step
        zpos += step
        move('z', zpos, ser)
        count += 1
        print("Count: ", count)
        #time.sleep(0.075)

        while xpos >= 0:
            move('x', xpos, ser)
            count += 1
            print("Count: ", count)
            x_voltages.append(xpos)
            z_voltages.append(zpos)
            powers.append(get_exposure(100))
            xpos -= step
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

def align_max_intensity(x_voltages, z_voltages, powers, file):
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

def run(ser, file):
    print("Starting scan")
    step = 5

    x_voltages, z_voltages, powers = scan(ser, step, file)

    intensity_plot(x_voltages, z_voltages, powers)

    align_max_intensity(x_voltages, z_voltages, powers, file)

    print("Scan done")
