import serial
from motion import move
import photodiode_in

def run(ser, pdn, ch, file):
    print("'q' returns to menu, 'ENTER' returns integrated PD. Enter command as [axis] [voltage], e.g. >>y 15.2.")
    while True:
        user_input = input(">> ").strip()
        if user_input.lower() == 'q':
            break
        try:
            axis, value = user_input.split()
            move(axis, float(value), ser, ch, file) 
        except Exception as e:
            photodiode_in.getPower(pdn, ch, file)
