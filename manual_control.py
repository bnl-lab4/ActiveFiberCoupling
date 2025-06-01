import serial
import time

def run():
    SERIAL_PORT = '/dev/ttyACM0'
    BAUD_RATE = 115200

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud.")
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        return

    def move(axis: str, voltage: float):
        if axis.lower() not in ['x', 'y', 'z']:
            print("Invalid axis. Choose 'x', 'y', or 'z'.")
            return
        command = f"{axis.lower()}voltage={voltage}\n"
        ser.write(command.encode())
        print(f"Sent: {command.strip()}")

    try:
        # Center axes
        move('x', 0.0)
        move('y', 0.0)
        move('z', 0.0)

        while True:
            print(f"Enter command, e.g. 'x 1.2' or 'z -3' or 'exit':")
            user_input = input(">>").strip()
            if user_input.lower() == 'exit':
                break
            try:
                axis, value = user_input.split()
                move(axis, float(value))
            except Exception as e:
                print(f"Invalid input: {e}")
    finally:
        ser.close()
        print("Serial connection closed.")
