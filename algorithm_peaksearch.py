import serial
import time

def run():
    print("Running Peak Search Alignment...")

    ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
    time.sleep(2)

    def move(axis, pos):
        command = f"{axis}voltage={pos}\n"
        ser.write(command.encode())
        time.sleep(0.1)

    # Example: sweep x axis from -1 to 1
    for x in [i * 0.1 for i in range(-10, 11)]:
        move('x', x)
        print(f"x={x}, simulated signal={(1 - abs(x)):.2f}")  # Replace with real photodetector readout

    ser.close()
    print("Peak Search done.")
