#!/usr/bin/python

#to do: change ser to ser0


import sys
import serial
import time
from motion import move
import numpy as np
import importlib

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
import algo_focal_estimator
import algo_PSO

SERIAL_PORT0 = '/dev/ttyACM0'
SERIAL_PORT1 = '/dev/ttyACM1'
BAUD_RATE = 115200

# Open serial connection
def initialize_serial():
    try:
        #ser0 = serial.Serial(port=SERIAL_PORT,
        #    baudrate=BAUD_RATE,
        #    bytesize=serial.EIGHTBITS,
        #    parity=serial.PARITY_NONE,
        #    stopbits=serial.STOPBITS_ONE,
        #    timeout=2,
        #    xonxoff=False,
        #    rtscts=False,
        #    dsrdtr=False,
        #)
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
    pdn = 500     #default number of times to integrate photodiode input ADC
   
   # For random, grid, hill clibing, cross search, one cross
    # section, three cross sections, and fitting algorithm:
    # They only work for 0 unless you manually change ser here
    # and change to (0,1) in photodiode_in

    try:
        while True:
            print("\nManual Control Options")
            print("0:        Ch 0 w/PD input")
            print("1:        Ch 1 w/PD input")
           # print("m:        Duplex w/o PD")
            print("center 0: Center axes of Ch 0")
            print("center 1: Center axes of Ch 1")
            print("zero 0:   V=0 all axes of Ch 0")
            print("zero 1:   V=0 all axes of Ch 1\n")

            print("Automatic Alignment Options")
            print("c0:       Coarse search for 0 (calculate function)")
            print("c1:       Coarse search for 1 (calculate function)")
            print("f0:       Fine search for 0 (continuous search function)")
            print("f1:       Fine search for 1 (continuous search function)\n")

            print("Experimental Options")
            print("r:        Random search algorithm")
            print("g:        Grid search algorithm")
            print("h:        Hill climbing algorithm")
            print("c:        Cross search algorithm")
            print("1c:       One cross section")
            print("3c:       Three cross sections")
            print("f:        Fitting algorithm")
            print("fe:       Focal estimator")
            print("pso:      Particle Swarm Optimization\n")

            print("pd:       Change photodiode integration count")
            print("reload:   Reload all algorithms and control modes")
            print("q to Quit\n")

            choice = input("> ").strip()

            #file = open("run_data.txt", "w")
            
            with open("new_file.txt", "a") as file:
                if choice == '0':
                    manual_control.run(ser0, pdn, 0, file)
                elif choice == '1':
                    manual_control.run(ser1, pdn, 1, file)
                elif choice == 'm':
                    '''
                    print("\nSelect a controller (0 or 1)")
                    new_choice = input("Enter your choice: ").strip()
                    if new_choice == '0':
                        manual_control.run(ser)
                    elif new_choice == '1':
                        manual_control.run(ser1)
                    else:
                        print("Invalid choice. Try again.")
                    '''
                    manual_control_duplex.run(ser0, ser1)
                elif choice == 'center 0':
                    center_axes.run(ser0, 0, file)
                    photodiode_in.getPower(pdn, 0, file)
                elif choice == 'center 1':
                    center_axes.run(ser1, 1, file)
                    photodiode_in.getPower(pdn, 0, file)
                elif choice == 'zero 0':
                    zero_axes.run(ser0, 0, file)
                    photodiode_in.getPower(pdn, 0, file)
                elif choice == 'zero 1':
                    zero_axes.run(ser1, 1, file)
                    photodiode_in.getPower(pdn, 0, file)
                elif choice == 'r':
                    algo_randomsearch.run(ser0, file)
                elif choice == 'g':
                    algo_gridsearch.run(ser0, file)
                elif choice == 'h':
                    algo_hill_climbing.run(ser0, file)
                elif choice == 'c':
                    algo_crossSearch.run(ser0)
                    #algo_crossSearch.run(ser1)
                elif choice == '1c':
                    algo_one_cross_section.run(ser0, file)
                elif choice == '3c':
                    algo_three_cross_sections.run(ser0, file)
                elif choice == 'c0':
                    #algo_calculate.run(ser, file, 0)
                    #data_algo_calculate.run(ser, file)
                    data_2_algo_calculate.run(ser0, file, 0, 0)
                elif choice == 'c1':
                    #algo_calculate.run(ser1, file, 1)
                    data_2_algo_calculate.run(ser1, file, 1, 1)
                elif choice == 'f0':
                    #algo_continuous_search.run(ser, file, 0)
                    #data_algo_continuous_search.run(ser, file)
                    data_2_algo_continuous_search.run(ser0, file, 0, 0)
                elif choice == 'f1':
                    #algo_continuous_search.run(ser1, file, 1)
                    data_2_algo_continuous_search.run(ser0, file, 1, 1)
                elif choice == 'f':
                    third_algo_one_cross_section_with_plots.run(ser0, file)
                elif choice == 'fe':
                    algo_focal_estimator.run(ser0, file)
                elif choice == 'pso':
                    algo_PSO.run(ser0,file,0,100)
                elif choice == 'pd':
                    choice = input("Enter PD integration count: ").strip()
                    pdn = choice.lower()
                    pdn = int(pdn)
                    print(f"New PD n-value = {pdn}")
                elif choice == 'reload':
                    for module in list(sys.modules.values()):
                        try:
                            #   print(f'{module.__file__.split("/")}\n')
                            if module.__file__.split('/')[4] == 'ActiveFiberCoupling':
                                importlib.reload(module)
                                print(f'{module.__name__} reloaded.')
                        except:
                            pass
                elif choice == 'q':
                    break
                else:
                    print("Invalid choice. Try again.")


    finally:
        ser0.close()
        ser1.close()
        print("Serial connection closed.")

if __name__ == '__main__':
    main()
