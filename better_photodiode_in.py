import socket
import photodiode_in as phd

#s = None

def getPower(time, *args, **kwargs):
    #    global s
    #print(time)
    #s.sendall(str(time).encode('utf-8'))
    #power = float(s.recv(1024).decode('utf-8'))
    #print(power)
    #return power
    return phd.getPower(time)

def get_exposure(time, *args, **kwargs):
    return getPower(time)
