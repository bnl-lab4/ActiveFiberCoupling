#!/home/bnl/ActiveFiberCoupling/venv/bin/python

import serial
import time
from motion import move
import numpy as np

#Device Imports
import piplates.DAQC2plate as DAQ

# Import alignment algorithms and control modes
import manual_control
import manual_keycontrol
import algo_randomsearch
import algo_gridsearch
import algo_hill_climbing
import algo_crossSearch
import algo_one_cross_section
import algo_three_cross_sections
import algo_calculate
import algo_cross_section_search
import algo_new_calculate

SERIAL_PORT = '/dev/ttyACM0'
#SERIAL_PORT1 = '/dev/ttyACM1'
BAUD_RATE = 115200

# Open serial connection
def initialize_serial():
    try:
        ser = serial.Serial(port=SERIAL_PORT,
            baudrate=BAUD_RATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=2,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        )
        #ser1 = serial.Serial(SERIAL_PORT1, BAUD_RATE, timeout=1)
        time.sleep(2)  # Give the device time to initialize
        print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud.")
        #print(f"Connected to {SERIAL_PORT1} at {BAUD_RATE} baud.")
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
            print("5. Hill climbing algorithm")
            print("6. Cross search algorithm")
            print("7. One cross section")
            print("8. Three cross sections")
            print("9. Calculate")
            print("10. Cross section search")
            print("11. New calculate")
            print("12. Exit")

            choice = input("Enter your choice: ").strip()

            #file = open("run_data.txt", "w")
            
            with open("data1.txt", "a") as file:
                if choice == '1':
                    manual_control.run(ser)
                elif choice == '2':
                    manual_keycontrol.run(ser)
                elif choice == '3':
                    algo_randomsearch.run(ser, file)
                elif choice == '4':
                    algo_gridsearch.run(ser, file)
                elif choice == '5':
                    algo_hill_climbing.run(ser, file)
                elif choice == '6':
                    algo_crossSearch.run(ser)
                    #algo_crossSearch.run(ser1)
                elif choice == '7':
                    algo_one_cross_section.run(ser, file)
                elif choice == '8':
                    algo_three_cross_sections.run(ser, file)
                elif choice == '9':
                    algo_calculate.run(ser, file)
                elif choice == '10':
                    algo_cross_section_search.run(ser, file)
                elif choice == '11':
                    algo_new_calculate.run(ser, file)
                elif choice == '12':
                    break
                else:
                    print("Invalid choice. Try again.")


    finally:
        ser.close()
        print("Serial connection closed.")

if __name__ == '__main__':
    main()
