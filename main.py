#!/home/bnl/ActiveFiberCoupling/venv/bin/python

import serial
import time
from motion import move

# Import alignment algorithms and control modes
import manual_control
import manual_keycontrol
import algo_randomsearch
import algo_gridsearch

SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 115200

# Open serial connection
def initialize_serial():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # Give the device time to initialize
        print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud.")
        return ser
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        exit(1)

def main():
    ser = initialize_serial()

    try:
        while True:
            print("\nSelect an option:")
            print("1. Manual control (typed input)")
            print("2. Manual key control (keystrokes -- not working)")
            print("3. Random search algorithm")
            print("4. Grid search algorithm")
            print("5. Exit")

            choice = input("Enter your choice: ").strip()

            if choice == '1':
                manual_control.run(ser)
            elif choice == '2':
                manual_keycontrol.run(ser)
            elif choice == '3':
                algo_randomsearch.run(ser)
            elif choice == '4':
                algo_gridsearch.run(ser)
            elif choice == '5':
                break
            else:
                print("Invalid choice. Try again.")
    finally:
        ser.close()
        print("Serial connection closed.")

if __name__ == '__main__':
    main()
