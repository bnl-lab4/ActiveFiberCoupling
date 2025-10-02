################ TODO
# CONTINUOUS SENSOR READOUT FUNCTON
# populate default kwargs dictionaries
# move logging funcs to their own script
# add general movement mode to manual control
# rewrite command line arg handling to be more flexible (dict?)
# convert main menu to YAML
#####################


import os
import sys
import importlib    # for reloading modules
import logging
import warnings
import contextlib   # better than nested 'with' statements
import readline     # enables input() to save history just by being loaded
import atexit       # for saving readline history file
import inspect      # checking kwargs match to function
import traceback    # show traceback in main menu
from typing import Optional, Union

# Import alignment algorithms and control modes
from MovementClasses import StageDevices, MovementType
from SensorClasses import Sensor, SensorType
from Distance import Distance
from SimulationClasses import SimulationSensor, SimulationStageDevices
import manual_control
import MovementUtils
import grid_search
import StringUtils
import HillClimb


# Device info constants
SOCKET0 = dict(host = '192.168.1.10', port = 8000, sensortype = SensorType.SOCKET)
SIPM0 = dict(addr = 0, channel = 1, sensortype = SensorType.SIPM)
SIPM1 = dict(addr = 0, channel = 2, sensortype = SensorType.SIPM)
PHOTODIODE0 = dict(addr = 0, channel = 1, sensortype = SensorType.PHOTODIODE)
PHOTODIODE1 = dict(addr = 0, channel = 2, sensortype = SensorType.PHOTODIODE)
SIMSENSOR_ASPH = dict(propagation_axis = 'y', focal_ratio = 4.0, angle_of_deviation = 0)
SIMSENSOR_LABTELE = dict(propagation_axis = 'y', focal_ratio = 28.0, angle_of_deviation = 0)
SIMSENSOR_SKYTELE = dict(propagation_axis = 'y', focal_ratio = 7.0, angle_of_deviation = 0)

SENSOR0 = PHOTODIODE0
SENSOR1 = SIMSENSOR_LABTELE

PIEZO_PORT0 = '/dev/ttyACM0'
PIEZO_PORT1 = '/dev/ttyACM1'
BAUD_RATE = 115200

STEPPER_DICT0 = dict(x = '00485175', y = '00485185', z = '00485159')
STEPPER_DICT1 = dict(x = None, y = None, z = None)


# custom warning format to remove the source line
def custom_formatwarning(message, category, filename, lineno, line=None):
    # Custom warning formatter that removes the source line.
    return f"{filename}-line{lineno}-{category.__name__} :: {message}\n"


# Initial logging/warning configuration
log = logging.getLogger(__name__)
logging.captureWarnings(True)
warnings.formatwarning = custom_formatwarning


def AcceptInputArgs(inputTuple, inputArgs):
    # inputTuple must be in correct order, should be refactored

    CommandArg_ValueDict = dict(t=True, y=True, true=True, f=False, n=False, false=False)

    for arg in inputArgs:
        if '-' not in arg:
            warnings.warn(f"Command-line argument {arg} not formatted properly (arg-bool)")
            continue
        arg, val = arg.strip().lower().split('-')
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
        if arg == 'consoleloglevel':
            try:
                inputTuple[5] = getattr(logging, val)
            except AttributeError:
                warnings.warn(f"Could not find log level {val}, default will be kept")
                continue
            inputTuple[5] = val
            continue
        if arg == 'loglevel':
            try:
                inputTuple[6] = getattr(logging, val)
            except AttributeError:
                warnings.warn(f"Could not find log level {val}, default will be kept")
                continue
            inputTuple[6] = val
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
                  filename: Optional[str] = None, console_log_level: Union[str, int, None] = None,
                              log_level: Union[str, int, None] = None):
    # Logging Defaults
    if log_to_console is None:
        log_to_console = True
    if log_to_file is None:
        log_to_file = True
    if filename is None:
        filename = './log_output.txt'

    if log_level is None:
        level = logging.DEBUG
    elif isinstance(log_level, str):
        # Map string to logging level
        level = getattr(logging, log_level.upper(), logging.INFO)
    elif log_level in [int(10 * i) for i in range(1, 6)]:
        level = log_level
    else:
        raise ValueError(f"log_level {log_level} is not in an acceptable form")

    if console_log_level is None or not log_to_console:
        console_level = logging.INFO    # INFO for now, WARNING during science use
    elif isinstance(console_log_level, str):
        # Map string to logging level
        console_level = getattr(logging, console_log_level.upper(), logging.INFO)
    elif console_log_level in [int(10 * i) for i in range(1, 6)]:
        console_level = console_log_level
    else:
        raise ValueError(f"console_log_level {console_log_level} is not in an acceptable form")

    standard_format = logging.Formatter("%(asctime)s-%(levelname)s-" +
                "%(module)s-line%(lineno)d-%(funcName)s :: %(message)s")
    caught_warning_format = logging.Formatter("%(asctime)s-WARNING(CAUGHT)-%(message)s")

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    warnings_logger = logging.getLogger('py.warnings')
    warnings_logger.setLevel(logging.WARNING)
    warnings_logger.propagate = False    # so root_logger does not also get caught warning logs

    # clears existing handlers to avoid duplicates
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    if warnings_logger.hasHandlers():
        warnings_logger.handlers.clear()

    # logic for which level done above, never truly off
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(standard_format)
    root_logger.addHandler(console_handler)

    console_handler_warnings = logging.StreamHandler()
    console_handler_warnings.setLevel(console_level)
    console_handler_warnings.setFormatter(caught_warning_format)
    warnings_logger.addHandler(console_handler_warnings)

    # Check and add file handling
    if log_to_file and filename:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(standard_format)
        root_logger.addHandler(file_handler)

        file_handler_warnings = logging.FileHandler(filename)
        file_handler_warnings.setFormatter(caught_warning_format)
        warnings_logger.addHandler(file_handler_warnings)

    if not (log_to_console or log_to_file):
        logging.disable(logging.CRITICAL)

    log_locations = []
    log_levels = []

    log_locations.append('console')
    log_levels.append(logging.getLevelName(console_level))
    if log_to_file:
        log_locations.append(filename)
        log_levels.append(logging.getLevelName(level))
    if len(log_locations) == 0:
        log.info("Logging to nowhere")
    else:
        for loc, level in zip(log_locations, log_levels):
            log.info(f"Logging to {loc} with level {level}")


def Update_Logging():
    SettingNames = ["Log to console", 'Log to file', 'Log filename', 'Console log level', 'Log level']
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

    while LoggingSettings['Log to console']: # console log level?
        user_input = input("Console log level? [debug/info/warning/error/critical/skip]: ").strip().lower()
        if user_input == 's' or user_input == 'skip':
            break
        try:
            LoggingSettings['Console log level'] = getattr(logging, user_input.upper())
            break
        except AttributeError:
            print(f"Could not find log level matching {user_input}")
        print(f"Could not update log level to {user_input}")

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
    print(f"Update default exposure time (currently {ExposureTime}):" +
        "(iterations for SiPMs or photodiodes, milliseconds for sockets)")
    user_input = input(">> ").strip()
    try:
        user_input = int(float(user_input))
    except ValueError:
        print(f"Input {user_input} cannot be converted to an int. ExposureTime will remain at {ExposureTime}")
        return False
    ExposureTime = user_input
    return True


def reload_menu(menu):
    for key in menu.keys():
        if key.startswith('_') or key == 'reload':
            continue    # ignore menu section headers and this functions menu entry

        # get name of module function comes from
        func_module_name = menu[key]['func'].__module__
        if func_module_name == '__main__':
            continue    # can't reload this script itself

        try:
            # get function name in its module
            func_name = menu[key]['func'].__name__

            # reload the module
            reloaded_module = importlib.reload(importlib.import_module(func_module_name))

            # get func from reloaded module
            new_func = getattr(reloaded_module, func_name)

            # redefine func in menu
            menu[key]['func'] = new_func

            log.info(f"{key} function reloaded succesfully")

        except (ImportError, AttributeError):
            log.warning(f"{key} function was not reloaded succesfully")


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
    # load input history file
    history_file = './.main_history'
    if os.path.exists(history_file):
        readline.read_history_file(history_file)
    readline.set_history_length(500)        # only save the last 100 entries
    # before stopping the script, save the input history to the file
    atexit.register(readline.write_history_file, history_file)
    # Default runtime variable
    AutoHome = True # home motors upon establishing connection
    RequireConnection = False # raise exception if device connections fail to establish
    ExposureTime = 200  # default exposure time (units are sensor dependent)
    LoggingSettings = [None] * 4

    if len(sys.argv) > 1:
        InputArgs = sys.argv[1:]
        AutoHome, RequireConnection, *LoggingSettings = AcceptInputArgs(
                [AutoHome, RequireConnection] + LoggingSettings, InputArgs)
        # print(f"Received input args: {AutoHome}, {RequireConnection}, {LoggingSettings}")

    setup_logging(*LoggingSettings)

    with contextlib.ExitStack() as stack:
        if "propogation_axis" not in SENSOR0:
            sensor0 = stack.enter_context(Sensor(SENSOR0, SENSOR0['sensortype']))
            stage0 = stack.enter_context(StageDevices('stage0', PIEZO_PORT0, STEPPER_DICT0,
                                        sensor = sensor0, require_connection = RequireConnection,
                                                 autohome = AutoHome))
        else:
            sensor0 = stack.enter_context(SimulationSensor(**SENSOR0))
            stage0 = stack.enter_context(SimulationStageDevices('simstage0', sensor = sensor0))

        if "propagation_axis" not in SENSOR1:
            sensor1 = stack.enter_context(Sensor(SENSOR1, SENSOR1['sensortype']))
            stage1 = stack.enter_context(StageDevices('stage1', PIEZO_PORT1, STEPPER_DICT1,
                                        sensor = sensor1, require_connection = RequireConnection,
                                                 autohome = AutoHome))

        else:
            sensor1 = stack.enter_context(SimulationSensor(**SENSOR1))
            stage1 = stack.enter_context(SimulationStageDevices('simstage1', sensor = sensor1))

        MENU_DICT = {
            '_manual_control' : "Manual Control Options",
            'manual' : {
                'text'   : 'Stage manual control',
                'func'   : manual_control.run,
                'args'   : {
                            '0' : (stage0, ExposureTime),
                            '1' : (stage1, ExposureTime),
                            },
                    },
            'center' : {
                'text'   : 'Center piezo or stepper axes',
                'func'   : MovementUtils.center,
                'args'   : {
                            '0p' : (stage0, MovementType.PIEZO),
                            '1p' : (stage1, MovementType.PIEZO),
                            '0s' : (stage0, MovementType.STEPPER),
                            '1s' : (stage1, MovementType.STEPPER),
                            },
                    },
            'zero' : {
                'text'   : 'Zero piezo or stepper axes',
                'func'   : MovementUtils.zero,
                'args'   : {
                            '0p' : (stage0, MovementType.PIEZO),
                            '1p' : (stage1, MovementType.PIEZO),
                            '0s' : (stage0, MovementType.STEPPER),
                            '1s' : (stage1, MovementType.STEPPER),
                            },
                    },
            'energize' : {
                'text'   : 'Energize steppers',
                'func'   : MovementUtils.energize,
                'args'   : {
                            '0' : (stage0,),
                            '1' : (stage1,),
                            },
                'kwargs' : {
                            'all' : dict(axes='all')
                            },
                    },
            'deenergize' : {
                'text'   : 'Deenergize steppers',
                'func'   : MovementUtils.deenergize,
                'args'   : {
                            '0' : (stage0,),
                            '1' : (stage1,),
                            },
                'kwargs' : {
                            'all' : dict(axes='all')
                            },
                    },
            'home' : {
                'text'   : 'Home steppers',
                'func'   : MovementUtils.home,
                'args'   : {
                            '0' : (stage0,),
                            '1' : (stage1,),
                            },
                'kwargs' : {
                            'all' : dict(axes='all')
                            },
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
                'kwargs' : {
                            'coarse' : dict(spacing = Distance(15, "volts"), plot=True, planes=3),
                            'fine'   : dict()
                            },
                    },
            'hillclimb' : {
                'text'   : 'Hill climbing with piezos or steppers',
                'func'   : HillClimb.run,
                'args'   : {
                            '0p' : (stage0, MovementType.PIEZO, ExposureTime),
                            '0s' : (stage0, MovementType.STEPPER, ExposureTime),
                            '1p' : (stage1, MovementType.PIEZO, ExposureTime),
                            '1s' : (stage1, MovementType.STEPPER, ExposureTime)
                            },
                'kwargs' : {},
                    },
            '_misc'      : "Miscillaneous",
            'reload'  : {
                'text'   : 'Reload all ActiveFiberCoupling modules',
                'func'   : None,
                'args'   : {},
                    },
            'texp'    : {
                'text'   : 'Change the default exposure time',
                'func'   : Update_ExposureTime,
                'args'   : {},
                    },
            'log'    : {
                'text'   : 'Change the logging settings',
                'func'   : Update_Logging,
                'args'   : {},
                    },
            'help'   : {
                'text'   : "Help with menu or with a function ('help <func_name>')",
                'func'   : StringUtils.menu_help,
                'args'   : {}
                    }
                }

        while True:
            print('\n'*4)   # for readability
            display_menu(MENU_DICT)
            print('')
            user_input = input(">> ").strip()
            print('\n')

            try:
                # special cases
                if user_input.lower() == 'q':
                    break
                if user_input.lower() == 'exec':    # hidden mode :), pls don't abuse
                    print("WARNING: IN EXEC MODE\nAnything typed here will be executed as python code.")
                    exec(input('exec >> '))
                    continue
                if user_input.lower() == 'reload':
                    reload_menu(MENU_DICT)
                    continue
                if user_input.lower() == 'uwu':
                    print('UwU')
                    continue

                user_input = user_input.strip().split(' ')
                if len(user_input) == 0:
                    print('\nNo input given')
                    continue
                if len(user_input) == 1:
                    misc_start = int(list(MENU_DICT.keys()).index('_misc')) + 1 # relies on ordering :/
                    if user_input[0].lower() in list(MENU_DICT.keys())[misc_start:]:  # if it is in miscellaneous
                        MENU_DICT[user_input[0].lower()]['func']()
                    else:
                        print('\nInvalid input: not enough space-separated arguments')
                    continue
                if user_input[0] not in MENU_DICT.keys() or user_input[0].startswith('_'):
                    print('\nInvalid input')
                    continue

                func_key = user_input[0].lower()
                func = MENU_DICT[func_key]['func']
                args_key = ''.join(sorted(user_input[1].lower()))   # sort to allow e.g. s1 or 1s

                if func_key in ('manual', 'home', 'energize', 'deenergize', 'status'):
                    # device-agnostic functions
                    args_key = args_key[0]

                if func_key == 'help':      # help is a special case, handled separately
                    StringUtils.menu_help(user_input[1], MENU_DICT)
                    continue

                args = MENU_DICT[func_key]['args'][args_key]
                if len(user_input) == 2:
                    func(*args)
                    continue

                user_input = user_input[2:]
                kwargs = {}
                if '=' not in user_input[0]:    # checking for default kwargs key
                    kwargs_default_key = user_input[0].lower()
                    kwargs.update(MENU_DICT[func_key]['kwargs'][kwargs_default_key])
                    user_input = user_input[1:]
                if any(['=' not in kwarg for kwarg in user_input]):
                    warnings.warn("All kwargs must be in key=value form")
                    print("Function call aborted.")
                    continue

                kwargs.update(StringUtils.str_to_dict(user_input))

                # check whether kwargs are correct
                try:
                    inspect.signature(func).bind(*args, **kwargs)
                except TypeError as e:
                    warnings.warn(f"Invalid keyword arguments: {e}")
                    print("Function call aborted.")
                    continue

                kwargs_print = StringUtils.dict_to_str(kwargs)
                print("Interpreted kwargs:\n" + kwargs_print)
                yn_input = input("Is this correct? (y/n): ").strip()
                if yn_input.lower() == 'y':
                    print(f"Calling {func_key} with args {args_key} and kwargs as above.\n")
                    func(*args, **kwargs)
                    continue
                elif yn_input.lower() == 'n':
                    print('')
                else:
                    print(f"Input {yn_input} could not be interpreted")

                print("Function call aborted.")

            except Exception as e:
                func_traceback = traceback.format_exc()
                warnings.warn(f"An error was encountered: {e}\nFull traceback below:\n" + func_traceback)


if __name__ == '__main__':
    main()
