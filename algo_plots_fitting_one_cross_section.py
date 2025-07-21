from motion import move
from photodiode_in import getPower
from photodiode_in import get_exposure
import time
import numpy as np
import matplotlib.pyplot as plt
import csv

import fitting.py

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

#If only one cross section works
def runone(ser, file):
    print("Starting scan")
    step = 5

    x_voltages, z_voltages, powers = scan(ser, step, file)

    #waists will hold the fitted waists of all cross sections
    #fill in wavelength of the laser, .635 = 635nm
    #fill in number of datapoints desired (total)
    waists = []
    wavelength = .635
    datapoints = 25
    #will try to map one cross section's data onto the gaussian function
    #for j in range(3):  #if uncommented, the loop will try to do it three times for each of the cross section but the code would change slightly, so I didn't do that for now
    waists.append(run(datapoints,wavelength,x_voltages,z_voltages,powers))
    
    #Following commented lines are for the runthree() function
    
    #fill in params with respective values, the w0 is the radii array
    #beamparams = waistfit(wavelength,waists)

    #second array item should be focal point ypos
    #focalpointy = beamparams[1]

    intensity_plot(x_voltages, z_voltages, powers)

    align_max_intensity(x_voltages, z_voltages, powers, file)

#If three cross sections work
def runthree(ser, file):
    print("Starting scan")
    step = 5

    x_voltages, z_voltages, powers = scan(ser, step, file)

    #waists will hold the fitted waists of all cross sections
    #fill in wavelength of the laser, .635 = 635nm
    #fill in number of datapoints desired (total)
    waists = []
    wavelength = .635
    datapoints = 25
    
    #will try to map one cross section's data onto the gaussian function
    for j in range(3):  #the loop will try to do it three times for each of the cross section but the code would change slightly because I didn't know how that was handled before
        waists.append(run(datapoints,wavelength,x_voltages,z_voltages,powers))
        x_voltages, z_voltages, powers = scan(ser, step, file)
    
    #fill in params with respective values, the w0 is the radii array
    beamparams = waistfit(wavelength,waists)

    #second array item should be focal point ypos
    focalpointy = beamparams[1]

    #unsure if I should put these lines in so commented. May be needed if alignment needs to happen after finding the focal point y

    #intensity_plot(x_voltages, z_voltages, powers)

    #align_max_intensity(x_voltages, z_voltages, powers, file)

    print("Scan done")
