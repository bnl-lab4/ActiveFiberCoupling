from motion import move
import time

def run(ser, file):
    print("Running grid search alignment algorithm...")
    step = 1
    for x in range(0, 76, step):
        for y in range(0, 76, step):
            for z in range(0, 76, step):
                move('x', x, ser, file)
                move('y', y, ser, file)
                move('z', z, ser, file)
                time.sleep(0.05)
    print("Grid search complete.")
