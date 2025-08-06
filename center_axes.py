from motion import move
import time, photodiode_in

def run(ser, ch, file):
    move('x', 37.5, ser, ch, file)
    move('y', 37.5, ser, ch, file)
    move('z', 37.5, ser, ch, file)
