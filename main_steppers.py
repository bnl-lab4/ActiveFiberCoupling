################ TODO
# logging control
#####################
#!/usr/bin/python


import sys
import serial
import time
from motion import move
import numpy as np
import importlib
import logging

#Device Imports
import piplates.DAQC2plate as DAQ

# Import alignment algorithms and control modes
from MovementClasses import StageDevices, MovementType
from SensorClasses import Sensor, SensorType
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

# Configure logging
# TODO custom formatter to truncate filepaths that are in ActiveFiberCoupling?
formatter = logging.Formatter("%(asctime)s - %(module)s - %(funcName)s :: %(message)s")
root_logger = logging.getLogger()
root_logger.setLevel(level=logging.INFO)        # set logging level
console_handler = logging.StreamHandler()       # show logging in console
console_handler.setFormatter(formatter)         #
root_logger.addHandler(console_handler)         #

# Device info
PHOTODIODE0 = dict(addr = 0, channel = 1)
PHOTODIODE1 = None

PIEZO_PORT0 = '/dev/ttyACM0'
PIEZO_PORT1 = '/dev/ttyACM1'
BAUD_RATE = 115200

STEPPER_DICT = dict(x=None, y=None, z=None)

ExposureTime = 500      # default number of times to integrate photodiode input ADC
                # in DAQ.getADC() function calls (approx 1ms?)


def reload_modules():
    print('\n')
    for module in list(sys.modules.values()):
        try:
            module_path = module.__file__
        except AttributeError:
            continue
        if module_path is None:
            continue
        if module.__file__ == '/home/bnl/ActiveFiberCoupling/main_steppers.py':
            # do not try to reload this file itself
            continue

        module_dirpath = ('/').join(module_path.split('/')[:-1])
        if module_dirpath == '/home/bnl/ActiveFiberCoupling':
            # if module is in this directory
            try:
                importlib.reload(module)
                print(f'Reloaded: {module.__name__}')
            except Exception as e:
                warnings.warn(f"RELOAD NOT SUCCESFUL: {module_dirpath}\n{e}")

def main():
    photodiode0 = Sensor(PHOTODIODE0, SensorType.PHOTODIODE)
    photodiode1 = None
#   sipm1 = Sensor(SIPM1, SensorType.SIPM)

    stage0 = StageDevices('stage0', PIEZO_PORT0, STEPPER_DICT,
                          sensor = photodiode0, require_connection = False)
    stage1 = StageDevices('stage1', PIEZO_PORT1, STEPPER_DICT,
                          sensor = photodiode1, require_connection = False)
   
    MENU_DICT = {
        '_manual_control' : "Manual Control Options",
        '0' : {
            'text'   : 'Stage 0 manual control',
            'func'   : manual_control.run,
            'args'   : (stage0, ExposureTime),
            'kwargs' : {}
                },
        '1' : {
            'text'   : 'Stage 1 manual control',
            'func'   : manual_control.run,
            'args'   : (stage1, ExposureTime),
            'kwargs' : {}
                },
        'center p0' : {
            'text'   : 'Center piezo axes of stage 0',
            'func'   : center_axes.run,
            'args'   : (stage0, MovementType.PIEZO),
            'kwargs' : {}
                },
        'center p1' : {
            'text'   : 'Center piezo axes of stage 1',
            'func'   : center_axes.run,
            'args'   : (stage0, MovementType.PIEZO),
            'kwargs' : {}
                },
        'center s0' : {
            'text'   : 'Center stepper axes of stage 0',
            'func'   : center_axes.run,
            'args'   : (stage0, MovementType.STEPPER),
            'kwargs' : {}
                },
        'center s1' : {
            'text'   : 'Center stepper axes of stage 1',
            'func'   : center_axes.run,
            'args'   : (stage0, MovementType.STEPPER),
            'kwargs' : {}
                },
        'zero p0' : {
            'text'   : 'Zero piezo axes of stage 0',
            'func'   : zero_axes.run,
            'args'   : (stage0, MovementType.PIEZO),
            'kwargs' : {}
                },
        'zero p1' : {
            'text'   : 'Zero piezo axes of stage 1',
            'func'   : zero_axes.run,
            'args'   : (stage0, MovementType.PIEZO),
            'kwargs' : {}
                },
        'zero s0' : {
            'text'   : 'Zero stepper axes of stage 0',
            'func'   : zero_axes.run,
            'args'   : (stage0, MovementType.STEPPER),
            'kwargs' : {}
                },
        'zero s1' : {
            'text'   : 'Zero stepper axes of stage 1',
            'func'   : zero_axes.run,
            'args'   : (stage0, MovementType.STEPPER),
            'kwargs' : {}
                },
        'reload'  : {
            'text'   : 'Reload all ActiveFiberCoupling modules',
            'func'   : reload_modules,
            'args'   : (),
            'kwargs' : {}
                }
            }

    def display_menu():
        max_choice_length = max(map(lambda s: len(s) if not s.startswith('_') else 0,
                                    list(MENU_DICT.keys())))    # excepting section titles, sorry
        whitespace = max_choice_length + 2    # for aligning descriptions
        for key, value in MENU_DICT.items():
            if key.startswith('_'):
                print('\n' + value)
            else:
                print(f"{key}:{' ' * (whitespace - len(key))}{value['text']}")
        return


    while True:
        display_menu() 
        user_input = input(">> ").strip()
        if user_input.lower() == 'q':
            break
        if user_input not in MENU_DICT.keys() or user_input.startswith('_'):
            print('\nInvalid input')
        else:
            MENU_DICT[user_input]['func'](*MENU_DICT[user_input]['args'],
                                          **MENU_DICT[user_input]['kwargs'])

if __name__ == '__main__':
    main()
