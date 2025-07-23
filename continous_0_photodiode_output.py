import numpy as np
import time

import piplates.DAQC2plate as DAQ

def print_avg_stdv():
    powers = []

    for i in range(10):
        p = getPower(10000)
        powers.append(p)

    print("Average:", np.average(powers))
    #print("STDV:", np.std(powers))

def get_power():
    power = []
    for i in range(100):
        power.append(DAQ.getADC(0,0))
    return round(((np.average(power)-0.016)/1.02),8)

def getPower(n):
    power = 0
    count = 0
    #print("Getting powers")
    for i in range(n):
        power+=DAQ.getADC(0,0)
        #power+=DAQ.getADC(0,1)
        count += 1
        #print(f"For loop iteration {count} done")
    power/=n
    #power = abs(power)
    return round(((power-0.016)/1.02),4)

def get_exposure(n):
    power = 0
    count = 0
    #print("Getting powers")
    for i in range(n):
        power+=DAQ.getADC(0,0)
        #power+=DAQ.getADC(0,1)
        count += 1
        #print(f"For loop iteration {count} done")
        #power = abs(power)
    return abs(round(((power-0.016)/1.02),4))

while True:
    power = get_exposure(100)
    print(f"Integration power: {power}")
