from motion import move
from photodiode_in import getPower
from photodiode_in import get_exposure
import time
import numpy as np
import matplotlib.pyplot as plt
import csv

import fittingvtwo

def coursescan(ser,file):
    intensity = []
    xpos = []
    distance = 0
    for i in range(50):
        input(f"Turn the course tuner to {distance % 5} (Left to Right) - Hit Enter to continue")
        intensity.append(get_exposure(100))
        distance += 1
        xpos.append(distance)

    print(f"here's the intensity array {intensity}")
    return intensity, xpos

def scan(ser, step, file):
    xpos = 0
    zpos = 0
    count = 0
    #x_voltages = np.array([])
    x_voltages = []
    z_voltages = []
    #z_voltages = np.array([])
    powers = []
    move('x', xpos, ser,0,file)
    move('z', zpos, ser,0,file)
    while zpos <= 74:
        while xpos <= 74:
            move('x', xpos, ser,0,file)
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
        move('z', zpos, ser,0,file)
        count += 15
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
    #xpos
    #second array item should be focal point ypos
    #focalpointy = beamparams[1]

    intensity_plot(x_voltages, z_voltages, powers)

    align_max_intensity(x_voltages, z_voltages, powers, file)

def findTopLine(zcenter,waists,ypos):
    scale = 20 / 75
    ypos = [y * scale for y in ypos]
    #z's have to be descending
    zmax = 1000
    ztop = []
    for j in range(3):
        if (zcenter[j] + waists[j]) < zmax:
            zmax = zcenter[j] + waists[j]
            ztop.append(zcenter[j] + waists[j])
        else:
            ztop.append(zcenter[j] - waists[j])
    #fit to x = my + b (first degree polynomial)
    coeff = np.polyfit(ypos,ztop,1)
    
    print(f"coeffone {coeff}")

    #return array with m,b
    return coeff
    
def findBottomLine(zcenter,waists,ypos):
    scale = 20 / 75
    ypos = [y * scale for y in ypos]
    #z's have to be ascending
    zmin = -1000
    zbot = []
    for j in range(3):
        if (zcenter[j] - waists[j]) > zmin:
            zmin = zcenter[j] - waists[j]
            zbot.append(zcenter[j] - waists[j])
        else:
            zbot.append(zcenter[j] + waists[j])
    #fit to z = my + b (first degree polynomial)
    coeff = np.polyfit(ypos,zbot,1)
    
    print(f"coefftwo {coeff}")

    #return array with m,b
    return coeff

def estimatefocal(coeffone,coefftwo):
    #take out m's and b's from the polyfit
    mtop,btop = coeffone
    mbot,bbot = coefftwo

    #find intersection point
    focalpointy = (bbot - btop) / (mtop - mbot)
    print(f"focal point y estimate: {focalpointy}")
    return focalpointy
    
    #estimate the zposition (70% sure this is useless because we're not fitting xdata
    #focalpointz = mtop * focalpointy + btop

def run(ser, file):
#def run(x_voltages,z_voltages,powers):
    '''
    coursearr = []
    x = 0
    while x < 50:
        coursearr.append(x)
        x += 1
    '''

    #oneintensity,coursearr = coursescan(ser,file)
    #oneintensity = [.6,.5088,.5627,.6049,.5951,.5843,.601,.5618,.5882,.5804,.5814,.5725,.6059,.5676,.6098,.6157,.6039,.602,.6,.6039,.6294,.6127,.6206,.6539,.3647,.3186,.3422,.3667,.399,.3549,.399,.401,.3657,.3373,.5686,17.8314,261.4412,453.2588,53.7216,.4941,.5186,.3667,.3843,.3176,.4304,.3392]   
    #popt = fittingvtwo.onegaussfit(coursearr,oneintensity)

    #input(f"Move the course tuner to {popt[1]}")

    print("Starting scan")
    step = 5
    ypos = [0,37.5,75]

    move('y',ypos[0],ser,0,file)
    x_voltages, z_voltages, powers = scan(ser, step, file)

    #waists will hold the fitted waists of all cross sections
    #fill in wavelength of the laser, 635 = 635nm
    waists = []
    xcenter = []
    zcenter = []
    wavelength = 635
    
    #will try to map one cross section's data onto the gaussian function
    for j in range(3):  #the loop will try to do it three times for each of the cross section but the code would change slightly because I didn't know how that was handled before
        optimized = fittingvtwo.run(wavelength,x_voltages,z_voltages,powers)
        print("fitting")
        xcenter.append(optimized[1])
        zcenter.append(optimized[2])
        waists.append(optimized[3])
        if j != 2:
            input("Press Enter to continue")
            move('y', ypos[j+1], ser,0,file)
            x_voltages, z_voltages, powers = scan(ser, step, file)
            #x_voltages = [x / 2 for x in x_voltages]
            #z_voltages = [z / 2 for z in z_voltages]
            
    coeffTop = findTopLine(zcenter,waists,ypos)
    coeffBot = findBottomLine(zcenter,waists,ypos)

    focalpointy = estimatefocal(coeffTop,coeffBot)
    

    #fill in params with respective values, the w0 is the radii array
    #beamparams = fittingvtwo.waistfit(wavelength,waists)

    #second array item should be focal point ypos
    #focalpointy = beamparams[1]
    print(f"change the yposition to: {focalpointy * 75/20}")
    
    move('y',focalpointy * 75/20,ser,0,file)
    input("Press enter to continue")

    x_voltages, z_voltages, powers = scan(ser,step,file)
    #unsure if I should put these lines in so commented. May be needed if alignment needs to happen after finding the focal point y

    optimized = fittingvtwo.run(wavelength,x_voltages,z_voltages,powers)
    print(f"x0 {optimized[1]}")
    print(f"z0 {optimized[2]}")
    print(f"A {optimized[0]}")

    move('x',optimized[1]*75/20,ser,0,file)
    move('z',optimized[2]*75/20,ser,0,file)
    move('y',focalpointy*75/20,ser,0,file)

    #intensity_plot(x_voltages, z_voltages, powers)

    #align_max_intensity(x_voltages, z_voltages, powers, file)

    print("Scan done")

'''
scale = 20 / 75

xpos = [
	0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,
	0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,
	0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,
	0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,
	0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,
	0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,
	0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,
	0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,
	0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,
	0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,
	0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,
	0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,
	0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,
	0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,
	0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,
]

#xpos = [x * scale for x in xpos]

zpos = []
for i in range(0, 71, 5):
    zpos.extend([i] * 15)

#zpos = [z * scale for z in zpos]
    
intensity_15x15 = [
    [0.6588, 0.6539, 0.6598, 0.648, 0.6461, 0.6422, 0.6441, 0.6569, 0.6304, 0.6324, 0.6373, 0.6157, 0.6248, 0.6206, 0.6333],
    [0.6549, 0.6539, 0.6598, 0.6382, 0.6431, 0.6304, 0.6402, 0.6167, 0.601,  0.6108, 0.6098, 0.6049, 0.6127, 0.6108, 0.6108],
    [0.6382, 0.6373, 0.648,  0.6451, 0.6343, 0.6235, 0.6167, 0.6108, 0.6127, 0.6167, 0.6265, 0.6294, 0.6069, 0.598,  0.5863],
    [0.6245, 0.65,   0.6392, 0.6451, 0.6147, 0.6294, 0.6108, 0.6216, 0.6225, 0.6275, 0.6314, 0.6441, 0.6382, 0.610,  0.5833],
    [0.6167, 0.6294, 0.6284, 0.6167, 0.6127, 0.5814, 0.5471, 0.4716, 0.3461, 0.2451, 0.2539, 0.2784, 0.4294, 0.5235, 0.5863],
    [0.6176, 0.6402, 0.6245, 0.602,  0.5637, 0.4637, 0.2431, 0.0196, 0.5196, 1.3598, 1.599,  1.5059, 1.2078, 0.5608, 0.1588],
    [0.5569, 0.6373, 0.6216, 0.5892, 0.4794, 0.2402, 0.1784, 0.9147, 2.0059, 3.6049, 4.3863, 4.5863, 3.8,    2.6451, 1.5373],
    [0.45,   0.651,  0.6176, 0.5608, 0.4843, 0.1637, 0.3098, 1.0765, 2.4098, 3.3598, 4.3676, 4.8235, 2.6559, 1.2843, 0.0804],
    [0.649,  0.6216, 0.5667, 0.499,  0.3735, 0.048,  0.3873, 1.1314, 1.8137, 2.2529, 2.2471, 0.6755, 0.8824, 0.002,  0.5745],
    [0.649,  0.6382, 0.6078, 0.5627, 0.4853, 0.4137, 0.2353, 0.0461, 0.0637, 0.0588, 0.0216, 0.1794, 0.4608, 0.6108, 0.6559],
    [0.6686, 0.6618, 0.6284, 0.6078, 0.5794, 0.5441, 0.4882, 0.4608, 0.4873, 0.4775, 0.5,    0.6059, 0.6588, 0.6618, 0.6637],
    [0.6608, 0.6569, 0.6539, 0.65,   0.649,  0.6108, 0.6049, 0.6059, 0.5863, 0.6,    0.593,  1.0,    0.6294, 0.6235, 0.6353],
    [0.648,  0.6588, 0.6686, 0.6559, 0.6627, 0.6539, 0.65,   0.6529, 0.652,  0.6402, 0.6471, 0.64,   0.02,   0.6353, 0.6412],
    [0.6343, 0.6588, 0.6657, 0.6716, 0.6716, 0.6696, 0.6667, 0.6618, 0.6716, 0.6559, 0.6696, 0.6686, 0.6657, 0.6627, 0.6725],5
    [0.6853, 0.6676, 0.6745, 0.6637, 0.6657, 0.6716, 0.6755, 0.6755, 0.6765, 0.6745, 0.6804, 0.66,   0.675,  0.673,  0.672]
]

intensity_15x15 = np.array(intensity_15x15)
intensity_15x15 = intensity_15x15.flatten()

#waists = []
#wavelength = 635
#for j in range(3):
#	waists.append(fittingtoo.run(wavelength,xpos,zpos,intensity_15x15)[3])
#	intensity = [i * .50 for i in intensity]

run(xpos,zpos,intensity_15x15)

#fill in params with respective values, the w0 is the radii array
#beamparams = fittingtoo.waistfit(wavelength,waists)

#second array item should be focal point ypos
#focalpointy = beamparams[1]
'''
