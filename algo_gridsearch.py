from motion import move
import time

def run(ser):
    print("Running grid search alignment algorithm...")
    step = 1
    for x in range(0, 76, step):
        for y in range(0, 76, step):
            for z in range(0, 76, step):
                move('x', x, ser)
                move('y', y, ser)
                move('z', z, ser)
                time.sleep(0.05)
    print("Grid search complete.")
