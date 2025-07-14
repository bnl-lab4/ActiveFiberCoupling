#General Imports
import time
import numpy as np

#Device Imports
import piplates.DAQC2plate as DAQ

#Functions
def get_power():
    power = []
    for i in range(100):
        power.append(DAQ.getADC(0,0))
    return round(((np.average(power)-0.016)/1.02),8)

def getPower(n):
    power = 0
    for i in range(n):
        power+=DAQ.getADC(0,0)
    power/=n
    return round(((power-0.016)/1.02),4)

if __name__ == "__main__":
    #Test Photodiode
    powers = []

    for i in range(100):
        p = getPower(100)
        powers.append(p)
        print(p)

    print("Average:", np.average(powers))
    print("STDV:", np.std(powers))
