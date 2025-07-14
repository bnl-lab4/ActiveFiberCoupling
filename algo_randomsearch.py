from motion import move
import time
import random

def run(ser, file):
    print("Running peak search alignment algorithm...")
    for i in range(5):
        for axis in ['x', 'y', 'z']:
            pos = random.uniform(0, 75)
            clamped = move(axis, pos, ser, file)
            print(f"Moved {axis.upper()} to {clamped:.3f}V\n")
            file.write(f"Moved {axis.upper()} to {clamped:.3f}V\n")
            time.sleep(0.05)
    print("Peak search complete.")
