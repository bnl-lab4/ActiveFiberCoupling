import socket

s = None
pdn = None

def getPower(time, *args, **kwargs):
    global s
    global pdn
    s.sendall(str(pdn).encode('utf-8'))
    power = float(s.recv(1024).decode('utf-8'))
#    print(power)
    return power

def get_exposure(time, *args, **kwargs):
    return getPower(time)
