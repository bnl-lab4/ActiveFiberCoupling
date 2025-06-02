from motion import move
import time
import random

def run(ser):
    print("Running peak search alignment algorithm...")
    for i in range(5):
        for axis in ['x', 'y', 'z']:
            pos = random.uniform(0, 75)
            move(axis, pos, ser)
            time.sleep(0.05)
    print("Peak search complete.")
