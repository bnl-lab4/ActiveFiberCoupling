################ TODO
# convert main menu to YAML
# reduce main menu entries by taking stage and device as inputs (e.g. p 0)
#   allow arbitrary kwarg entries with choices
#   kwargs in YAML will be defaults
#####################


import os
import sys
import importlib
import logging
import warnings
import contextlib
from typing import Optional, Union, Dict

# Import alignment algorithms and control modes
from MovementClasses import StageDevices, MovementType, Distance
from SensorClasses import Sensor, SensorType
import manual_control
import center_axes
import zero_axes
import grid_search


# unique logger name for this module
log = logging.getLogger(__name__)

# Device info constants
SOCKET0 = dict(host = '192.168.1.10', port = 8000, sensortype = SensorType.SOCKET)
SIPM0 = dict(addr = 0, channel = 1, sensortype = SensorType.SIPM)
SIPM1 = dict(addr = 0, channel = 2, sensortype = SensorType.SIPM)
PHOTODIODE0 = dict(addr = 0, channel = 1, sensortype = SensorType.PHOTODIODE)
PHOTODIODE1 = dict(addr = 0, channel = 2, sensortype = SensorType.PHOTODIODE)

SENSOR0 = PHOTODIODE0
SENSOR1 = PHOTODIODE1

PIEZO_PORT0 = '/dev/ttyACM0'
PIEZO_PORT1 = '/dev/ttyACM1'
BAUD_RATE = 115200

STEPPER_DICT0 = dict(x = '00485175', y = '00485185', z = '00485159')
STEPPER_DICT1 = dict(x = None, y = None, z = None)


def AcceptInputArgs(inputTuple, inputArgs):
    # inputTuple must be in correct order, should be refactored

    CommandArg_ValueDict = dict(t=True, y=True, true=True, f=False, n=False, false=False)

    for arg in inputArgs:
        print(arg)
        try:
            arg, val = arg.strip().lower().split('-')
        except Exception as e:
            print(e)
            warnings.warn(f"Command-line argument {arg} not formatted properly")
            continue
        if arg == 'autohome':
            inputTuple[0] = CommandArg_ValueDict[val]
            continue
        if arg == "requireconnection" or arg == 'requirecon':
            inputTuple[1] = CommandArg_ValueDict[val]
            continue
        if arg == 'logtoconsole':
            inputTuple[2] = CommandArg_ValueDict[val]
            continue
        if arg == 'logtofile':
            inputTuple[3] = CommandArg_ValueDict[val]
            continue
        if arg == 'logfile':
            if verify_logfile(val):
                inputTuple[4] = val
        if arg == 'loglevel':
            try:
                inputTuple[5] = getattr(logging, val)
            except AttributeError:
                warnings.warn(f"Could not find log level {val}, default will be kept")
                continue
            inputTuple[5] = val
            continue

        warnings.warn(f"{'-'.join(arg)} is not a valid argument")

    return inputTuple


def verify_logfile(filepath):
    try:
        # Check if the file exists
        if not os.path.exists(filepath):
            # If the file does not exist, create it
            # The 'x' mode is used for exclusive creation, failing if the file already exists.
            # This is a safe way to create a new file and handle potential race conditions.
            with open(filepath, 'x'):
                pass  # Do nothing, just create the file
            print(f"Log file created successfully at: {filepath}")
            return True
        else:
            print(f"Log file already exists at: {filepath}")
            return True
    except FileExistsError:
        # This is a good practice to handle a rare race condition where a file
        # is created by another process between the 'os.path.exists' check and 'open' call.
        print(f"Log file already exists at: {filepath}")
        return True
    except OSError as e:
        # Catch other OS-related errors, such as an invalid file path or permissions issues
        warnings.warn(f"Error: Invalid file path or other OS error: {e}\nUsing default log file")
        return False


def setup_logging(log_to_console: Optional[bool] = None, log_to_file: Optional[bool] = None,
                              filename: Optional[str] = None,
                              log_level: Union[str, int, None] = None):
    # Logging Defaults
    if log_to_console is None:
        log_to_console = True
    if log_to_file is None:
        log_to_file = True
    if filename is None:
        filename = './log_output.txt'
    if log_level is None:
        level = logging.INFO
    elif isinstance(log_level, str):
        # Map string to logging level
        level = getattr(logging, log_level.upper(), logging.INFO)
    else:
        level = log_level # it should be an int

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(module)s - %(funcName)s :: %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # clears existing handlers to avoid duplicates
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

    log_locations = []
    if log_to_console:
        log_locations.append('console')
    if log_to_file:
        log_locations.append(filename)
    if len(log_locations) == 0:
        log_locations = 'nowhere'
    else:
        log_locations = ' and '.join(log_locations)

    log.info(f"Logging to {log_locations} with level {logging.getLevelName(level)}")


def Update_Logging():
    SettingNames = ["Log to console", 'Log to file', 'Log filename', 'Log level']
    LoggingSettings = {name : None for name in SettingNames}

    while True: # Log to console?
        user_input = input("Log to console? [y/n/s]: ").strip().lower()
        if user_input == 's':
            break
        if user_input == 'y':
            LoggingSettings['Log to console'] = True
            break
        if user_input == 'n':
            LoggingSettings['Log to console'] = False
            break
        print(f"Could not interpret {user_input}")

    while True: # log to file?
        user_input = input("Log to file? [y/n/s]: ").strip().lower()
        if user_input == 's':
            break
        if user_input == 'y':
            LoggingSettings['Log to file'] = True
            break
        if user_input == 'n':
            LoggingSettings['Log to file'] = False
            break
        print(f"Could not interpret {user_input}")

    while LoggingSettings['Log to file']: # log file?
        user_input = input("Log filename? [s]: ").strip()
        if user_input == 's':
            break
        try:
            if verify_logfile(user_input):
                LoggingSettings['Log filename'] = user_input
                break
        except Exception as e:
            print(f"Error while verifying input filename: {e}")
        print(f"Could not update log file to {user_input}")

    while True: # log level?
        user_input = input("Log level? [debug/info/warning/error/critical/skip]: ").strip().lower()
        if user_input == 's' or user_input == 'skip':
            break
        try:
            LoggingSettings['Log level'] = getattr(logging, user_input.upper())
            break
        except AttributeError:
            print(f"Could not find log level matching {user_input}")
        print(f"Could not update log level to {user_input}")

    setup_logging(*list(LoggingSettings.values()))


def Update_ExposureTime():
    global ExposureTime
    print("Update default exposure time (iterations for SiPMs or photodiodes, milliseconds for sockets)")
    user_input = input(">> ").strip()
    try:
        user_input = int(float(user_input))
    except ValueError:
        print("Input {user_input} cannot be converted to an int. ExposureTime will remain at {ExposureTime}")
        return False
    ExposureTime = user_input
    return True


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


def display_menu(menu_dict):
    max_choice_length = max(map(lambda s: len(s) if not s.startswith('_') else 0,
                                list(menu_dict.keys())))    # excepting section titles, sorry
    whitespace = max_choice_length + 2    # for aligning descriptions
    for key, value in menu_dict.items():
        if key.startswith('_'):
            print('\n' + value)
        else:
            print(f"{key}:{' ' * (whitespace - len(key))}{value['text']}")
    return


def main():
    # Default runtime variable
    AutoHome = True # home motors upon establishing connection
    RequireConnection = False # raise exception if device connections fail to establish
    ExposureTime = 200  # default exposure time (units are sensor dependent)
    LoggingSettings = [None] * 4

    if len(sys.argv) > 1:
        InputArgs = sys.argv[1:]
        AutoHome, RequireConnection, *LoggingSettings = AcceptInputArgs(
                [AutoHome, RequireConnection] + LoggingSettings, InputArgs)
        print(f"Received input args: {AutoHome}, {RequireConnection}, {LoggingSettings}")

    setup_logging(*LoggingSettings)

    with contextlib.ExitStack() as stack:
        sensor0 = stack.enter_context(Sensor(SENSOR0, SENSOR0['sensortype']))
        sensor1 = stack.enter_context(Sensor(SENSOR1, SENSOR1['sensortype']))

        stage0 = stack.enter_context(StageDevices('stage0', PIEZO_PORT0, STEPPER_DICT0,
                                        sensor = sensor0, require_connection = RequireConnection,
                                                 autohome = AutoHome))
        stage1 = stack.enter_context(StageDevices('stage1', PIEZO_PORT1, STEPPER_DICT1,
                                        sensor = sensor1, require_connection = RequireConnection,
                                                 autohome = AutoHome))

        MENU_DICT = {
            '_manual_control' : "Manual Control Options",
            'manual' : {
                'text'   : 'Stage manual control',
                'func'   : manual_control.run,
                'args'   : {
                            '0' : (stage0, ExposureTime),
                            '1' : (stage1, ExposureTime),
                            },
                'kwargs' : {}
                    },
            'center' : {
                'text'   : 'Center piezo or stepper axes',
                'func'   : center_axes.run,
                'args'   : {
                            '0p' : (stage0, MovementType.PIEZO),
                            '1p' : (stage1, MovementType.PIEZO),
                            '0s' : (stage0, MovementType.STEPPER),
                            '1s' : (stage1, MovementType.STEPPER),
                            },
                'kwargs' : {}
                    },
            'zero' : {
                'text'   : 'Zero piezo or stepper axes',
                'func'   : zero_axes.run,
                'args'   : {
                            '0p' : (stage0, MovementType.PIEZO),
                            '1p' : (stage1, MovementType.PIEZO),
                            '0s' : (stage0, MovementType.STEPPER),
                            '1s' : (stage1, MovementType.STEPPER),
                            },
                'kwargs' : {}
                    },
            '_optimization' : 'Optimization Algorithms',
            'grid'    : {
                'text'   : 'Grid search with piezos or steppers',
                'func'   : grid_search.run,
                'args'   : {
                            '0p' : (stage0, MovementType.PIEZO, ExposureTime),
                            '1p' : (stage1, MovementType.PIEZO, ExposureTime),
                            '0s' : (stage0, MovementType.STEPPER, ExposureTime),
                            '1s' : (stage1, MovementType.STEPPER, ExposureTime),
                            },
                'kwargs' : dict(spacing = Distance(15, "volts"), plot=True, planes=3)
                    },
            '_misc'      : "Miscillaneous",
            'reload'  : {
                'text'   : 'Reload all ActiveFiberCoupling modules (might be broken)',
                'func'   : reload_modules,
                'args'   : (),
                'kwargs' : {}
                    },
            'texp'    : {
                'text'   : 'Change the default exposure time',
                'func'   : Update_ExposureTime,
                'args'   : (),
                'kwargs' : {}
                    },
            'log'    : {
                'text'   : 'Change the logging settings',
                'func'   : Update_Logging,
                'args'   : (),
                'kwargs' : {}
                    },
                }

        while True:
            display_menu(MENU_DICT)
            user_input = input(">> ").strip().lower()
            if user_input == 'q':
                break
            user_input = user_input.split()
            if len(user_input) == 0:
                print('\nNo input given')
                continue
            if len(user_input) == 1:
                if user_input[0] in list(MENU_DICT.keys())[-3:]:  # if it is in miscellaneous
                    MENU_DICT[user_input[0]]['func']()
                else:
                    print('\nInvalid input: not enough space-separated arguments')
                continue
            if user_input[0] not in MENU_DICT.keys() or user_input[0].startswith('_'):
                print('\nInvalid input')
                continue

            MENU_DICT[user_input[0]]['func'](*MENU_DICT[user_input[0]]['args'][user_input[1]],
                                          **MENU_DICT[user_input[0]]['kwargs'])


if __name__ == '__main__':
    main()
