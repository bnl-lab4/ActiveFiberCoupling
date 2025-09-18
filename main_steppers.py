################ TODO
# convert main menu to YAML
# reduce main menu entried by taking stage and device as inputs (e.g. p0)
#   allow arbitrary kwarg entries with choices
#   kwargs in YAML will be defaults
#####################
#!/usr/bin/python


import sys
import serial
import time
from motion import move
import numpy as np
import importlib
import logging
import warnings
import contextlib

#Device Imports
import piplates.DAQC2plate as DAQ

# Import alignment algorithms and control modes
from MovementClasses import StageDevices, MovementType, Distance
from SensorClasses import Sensor, SensorType
import manual_control
import center_axes
import zero_axes
import grid_search

# Configure logging
LOG_TO_CONSOLE = True # toggle to log to console
LOG_TO_FILE = True # toggle to log to file
LOG_FILENAME = "./log_output.txt" #need filepath
LOG_LEVEL = "INFO" # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

def setup_logging(log_to_console: bool = LOG_TO_CONSOLE, log_to_file: bool = LOG_TO_FILE, filename = LOG_FILENAME, log_level: str = LOG_LEVEL):
	
	#Map string to logging level
	level = getattr(logging, log_level.upper(), logging.INFO)
	formatter - logging.formatter("%(asctime)s - %(module)s - %(funcName)s :: %(message)s")
	root_logger = logging.getLogger()
	root_logger.setLevel(level)

	#clears existing handlers to avoid duplicates
	if root_logger.hasHandlers():
		root_logger.handlers.clear()

	# Check and add console logging
	if log_to_console:
		console_handler = logging.StreamHandler()
		console_handler.setFormatter(formatter)
		root_logger.addHandler(console_handler)

	# Check and add file handling
	if log_to_file and filename:
		file_handler = logging.FileHandler(filename)
		file_handler.setFormatter(formatter)
		root_logger.addHandler(file_handler)
	
	if not (log_to_console or log_to_file):
		logging.disable(logging.CRITICAL)


# Device info
PHOTODIODE0 = dict(addr = 0, channel = 1)
PHOTODIODE1 = None

PIEZO_PORT0 = '/dev/ttyACM0'
PIEZO_PORT1 = '/dev/ttyACM1'
BAUD_RATE = 115200

STEPPER_DICT0 = dict(x = '00485185', y = '00485159', z = '00485175')
STEPPER_DICT1 = dict(x = None, y = None, z = None)

ExposureTime = 500      # default number of times to integrate photodiode input ADC
                # in DAQ.getADC() function calls (approx 1ms?)


def reload_modules():       #   not working?
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
    with contextlib.ExitStack() as stack:
        photodiode0 = stack.enter_context(Sensor(PHOTODIODE0, SensorType.PHOTODIODE))
        photodiode1 = None
    #   sipm1 = Sensor(SIPM1, SensorType.SIPM)

        stage0 = stack.enter_context(StageDevices('stage0', PIEZO_PORT0, STEPPER_DICT0,
                              sensor = photodiode0, require_connection = False))
        stage1 = stack.enter_context(StageDevices('stage1', PIEZO_PORT1, STEPPER_DICT1,
                              sensor = photodiode1, require_connection = False))
       
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
                'text'   : 'Reload all ActiveFiberCoupling modules (might be broken)',
                'func'   : reload_modules,
                'args'   : (),
                'kwargs' : {}
                    },
            '_optimization' : 'Optimization Algorithms',
            'grid p0'    : {
                'text'   : 'Grid search with piezos of stage 0',
                'func'   : grid_search.run,
                'args'   : (stage0, MovementType.PIEZO, ExposureTime),
                'kwargs' : dict(spacing = Distance(15, "volts"), plot=False, planes=3)
                    },
            'grid p1'    : {
                'text'   : 'Grid search with piezos of stage 1',
                'func'   : grid_search.run,
                'args'   : (stage1, MovementType.PIEZO, ExposureTime),
                'kwargs' : {}
                    },
            'grid s0'    : {
                'text'   : 'Grid search with steppers of stage 0',
                'func'   : grid_search.run,
                'args'   : (stage0, MovementType.STEPPER, ExposureTime),
                'kwargs' : {}
                    },
            'grid s1'    : {
                'text'   : 'Grid search with steppers of stage 1',
                'func'   : grid_search.run,
                'args'   : (stage1, MovementType.STEPPER, ExposureTime),
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
