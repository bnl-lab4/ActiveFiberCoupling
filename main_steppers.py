#!/usr/bin/python


import sys
import serial
import time
from motion import move
import numpy as np
import importlib

#Device Imports
import piplates.DAQC2plate as DAQ

# Import alignment algorithms and control modes
from MovementClasses import StageDevices
import main_menu
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
import warnings

SERIAL_PORT0 = '/dev/ttyACM0'
SERIAL_PORT1 = '/dev/ttyACM1'
BAUD_RATE = 115200

STEPPER_DICT = dict(x=None, y=None, z=None)


def display_menu():
    max_choice_length = max(map(len, list(MENU_DICT.keys())))
    whitespace = max_choice_length + 2
    for key, value in MENU_DICT.items():
        if key.startswith('_'):
            print('\n' + value)
        else:
            print(f"{key}:{' ' * whitespace}{value[0]}")
    return


def main():
    stage0 = StageDevices('stage0', SERIAL_PORT0, STEPPER_DICT, require_connection=False)
    stage1 = StageDevices('stage1', SERIAL_PORT1, STEPPER_DICT, require_connection=False)
    Texp = 500      # in ms
                    # default number of times to integrate photodiode input ADC
   
    MENU_DICT = {
        '_manual_control' : "Manual Control Options",
        '0' : {
            'text'   : 'Stage 0 manual control',
            'func'   : manual_control.run,
            'args'   : (stage0, Texp),
            'kwargs' : {}
                },
        '1' : {
            'text'   : 'Stage 1 manual control',
            'func'   : manual_control.run,
            'args'   : (stage1, Texp),
            'kwargs' : {}
                },
        'center p0' : ('Center piezo axes of stage 0', ),
        'center p1' : ('Center piezo axes of stage 1', ),
        'center s0' : ('Center stepper axes of stage 0', ),
        'center s1' : ('Center stepper axes of stage 1', ),
        'zero p0' : ('Zero piezo axes of stage 0', ),
        'zero p1' : ('Zero piezo axes of stage 1', ),
        'zero s0' : ('Zero stepper axes of stage 0', ),
        'zero s1' : ('Zero stepper axes of stage 1', ),
            }

   # For random, grid, hill clibing, cross search, one cross
    # section, three cross sections, and fitting algorithm:
    # They only work for 0 unless you manually change ser here
    # and change to (0,1) in photodiode_in

    while True:
        main_menu.display_menu() 
        user_input = input(">> ").strip()
        if user_input.lower() == 'q':
            break
        if user_input not in MENU_DICT.keys() or user_input.startswith('_'):
            print('Invalid input')
        else:
            MENU_DICT[user_input]['func'](*MENU_DICT[user_input]['args'],
                                          **MENU_DICT[user_input]['kwargs'])

if __name__ == '__main__':
    main()
