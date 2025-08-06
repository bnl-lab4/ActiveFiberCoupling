#!/usr/bin/python

#to do: change ser to ser0


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
import algo_continuous_search
import third_algo_one_cross_section_with_plots
import data_algo_calculate
import data_algo_continuous_search
import data_2_algo_calculate
import data_2_algo_continuous_search
import center_axes
import zero_axes
import photodiode_in

SERIAL_PORT0 = '/dev/ttyACM1'
SERIAL_PORT1 = '/dev/ttyACM0'
BAUD_RATE = 115200

# Open serial connection
def initialize_serial():
    try:
        ser0 = serial.Serial(SERIAL_PORT0, BAUD_RATE, timeout=1)
        ser1 = serial.Serial(SERIAL_PORT1, BAUD_RATE, timeout=1)
        time.sleep(2)  # Give the device time to initialize
        print(f"Connected to {SERIAL_PORT0} at {BAUD_RATE} baud.")
        print(f"Connected to {SERIAL_PORT1} at {BAUD_RATE} baud.")
        return ser0, ser1
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        exit(1)

def main():
    ser0, ser1 = initialize_serial()
    pdn = 50000     #default number of times to integrate photodiode input ADC
    samples = 10

    try:
        for i in range(samples):
            with open("chara_data.txt", "a") as file:
                    center_axes.run(ser0, 0, file)
                    photodiode_in.getPower(pdn, 0, file)
                    zero_axes.run(ser0, 0, file)
                    photodiode_in.getPower(pdn, 0, file)

    finally:
        ser0.close()
        ser1.close()
        print("Serial connection closed.")

if __name__ == '__main__':
    main()
