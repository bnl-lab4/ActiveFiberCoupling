from motion import move
import time, photodiode_in

def run(ser, ch, file):
    move('x', 0, ser, ch, file)
    move('y', 0, ser, ch, file)
    move('z', 0, ser, ch, file)
