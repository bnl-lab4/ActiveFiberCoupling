################ TODO
# convert main menu to YAML
# create simulator StageDevice with fake data
# add unit input to manual control
# add center argument back into grid_search
# add get piezo/stepper position function to StageAxis
# 3d fitting in grid_search
# populate default kwargs dictionaries
#####################


import os
import sys
import importlib    # for reloading modules
import logging
import warnings
import shlex        # better parsing of input kwargs
import contextlib   # better than nested 'with' statements
import readline     # enables input() to save history just by being loaded
import atexit       # for saving readline history file
import inspect      # checking kwargs match to function
import traceback    # show traceback in main menu
from typing import Optional, Union, List, Sequence

# Import alignment algorithms and control modes
from MovementClasses import StageDevices, MovementType, Distance
from SensorClasses import Sensor, SensorType
import manual_control
import center_axes
import zero_axes
import MovementUtils
import grid_search


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


def parse_str_values(value):
    # Attempt to convert to appropriate data type
    if value.lower() in ['true', 'false']:
        return value.lower() == 'true'

    if value.startswith('[') and value.endswith(']'):
        # Check if value is list
        value = value[1:-1]
        list_values = [item.strip() for item in value.split(',')]
        for i, val in enumerate(list_values):
            if val.startswith('D('):
                list_values[i] = list_values[i] + ',' + list_values[i+1]
                del list_values[i+1]
        value = [parse_str_values(value) for value in list_values]
        return value

    if (value.startswith('D(') or value.startswith('Distance(')) and \
                        value.endswith(')'):
        # Check if value is meant to be a Distance object
        value = value[2:-1]
        value = [item.strip() for item in value.split(',')]
        distance_args = [parse_str_values(arg) for arg in value]
        return Distance(*distance_args)

    if value.replace('.', '', 1).isdigit():
        # Check for float or integer
        if '.' in value:
            return float(value)
        return int(value)

    if (value.startswith('"') and value.endswith('"')) or \
         (value.startswith("'") and value.endswith("'")):
        # Check for quotes indicating a string
        return value[1:-1]

    # Default to string if no other type matches
    return value


def str_to_dict(tokens: List[str]):     # expects tokens as shlex would return
    if isinstance(tokens, str):
        tokens = [tokens,]
    assert isinstance(tokens, list), "tokens must be a string or list thereof"

    kwargs_dict = {}
    rejects = []
    for pair in tokens:
        if '=' in pair:     # require '=' to separate key and value
            key, value = [part.strip() for part in pair.split('=', 1)]
            value = parse_str_values(value)
            kwargs_dict[key] = value
        else:
            rejects.append(pair)
    if len(rejects) != 0:
        warnings.warn("The following input kwargs could not be parsed:" +
                      '\n'.join(rejects))

    return kwargs_dict


def sequence_to_str(sequence, joined = True):
    sequence_print = []
    for elem in sequence:
        if isinstance(elem, dict):
            sub_dict_list = dict_to_str(elem, joined=False)
            if len(sub_dict_list) <= 1:
                sub_dict_print = '{ ' + sub_dict_list[0] + ' }'
            else:
                sub_dict_print = "{\n" + '\n'.join(sub_dict_list) + "\n}"
            sub_dict_print = sub_dict_print.replace('\n', '\n' + ' ' * 8)
            sequence_print.append(sub_dict_print)
            continue

        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            sub_seq_list = sequence_to_str(elem, joined=False)
            if len(sub_seq_list) <= 1:
                sub_seq_print = '( ' + sub_seq_list + " )"
            else:
                sub_seq_print = '(\n' + '\n'.join(sub_seq_list) + '\n)'
            sub_seq_print = sub_seq_print.replace('\n', '\n' + ' ' * 8)
            sequence_print.append(sub_seq_print)
            continue

        elif isinstance(elem, Distance):
            sequence_print.append(elem.prettyprint())
        else:
            sequence_print.append(str(elem))

    if joined:
        sequence_print = '\n'.join(sequence_print)
    return sequence_print


def dict_to_str(mydict, joined = True):
    dict_print = []
    for key, value in mydict.items():
        if isinstance(value, dict):
            sub_dict_list = dict_to_str(value, joined=False)
            if len(sub_dict_list) <= 1:
                sub_dict_print = f"{str(key)} : " + '{ ' + sub_dict_list[0] + ' }'
            else:
                sub_dict_print = f"{str(key)} : " + '{\n' + '\n'.join(sub_dict_list) + "\n}"
            sub_dict_print = sub_dict_print.replace('\n', '\n' + ' ' * 8)
            dict_print.append(sub_dict_print)
            continue

        elif isinstance(value, Sequence) and not isinstance(value, str):
            sub_seq_list = sequence_to_str(value, joined=False)
            if len(sub_seq_list) <= 1:
                sub_seq_print = f"{str(key)} : ( {sub_seq_list[0]} )"
            else:
                sub_seq_print = f"{str(key)} : (\n" + '\n'.join(sub_seq_list) + '\n)'
            sub_seq_print = sub_seq_print.replace('\n', '\n' + ' ' * 8)
            dict_print.append(sub_seq_print)
            continue

        elif isinstance(value, Distance):
            dict_print.append(value.prettyprint())
        else:
            dict_print.append(f"{str(key)} : {str(value)}")

    if joined:
        dict_print = '\n'.join(dict_print)
    return dict_print


def menu_help(func_key: Optional[str] = None, menu: Optional[dict] = None):
    MAIN_MENU_HELP = """
MAIN MENU HELP
Function call syntax is '<func name> <stagenum device> <default kwarg name> <key=value> <key=value>'.
Call 'help func=<func name>' to see the required args and optional kwargs.
Space is the delimiter between arguments, so do not use spaces anywhere else.
Keyword argument values can be ints, floats, strings, Distance objects, and lists thereof.
Strings are handled lazily and do not need to be wrapped with ' or " (but can be).
Lists are denoted by starting and ending with brackets '[' ']', with the elements comma separated.
Distance objects are denoted by starting with 'D(' or 'Distance(' and ending with parentheses ')'.
The first argument of Distance is the value, the second is the units, separated by only a comma.

Some examples:
help func=reload
        -- Show the input parameters of reload.
        -- Certain functions like help don't need an arg.
center 0s
        -- Centers the steppers of stage 0.
grid 0s fine axes='yz' planes=[D(100,"fullsteps"),D(500,fullsteps)]
        -- Run grid search on stage 0 with steppers in y-z planes,
            using the 'fine' preset kwargs but overriding the planes argument.
    """

    if func_key is not None:
        func_dict_str = dict_to_str(menu[func_key])
        func_sig = inspect.signature(menu[func_key]['func'])
        siglist = []
        for _, sig in list(func_sig.parameters.items()):
            siglist.append(str(sig))
        sig_str = ',\n'.join(siglist)
        return print(f"FUNCTION MENU ENTRY:\n{func_dict_str}" + '\n'*2 +
                     f"FUNCTION SIGNATURE:\n{sig_str}")

    return print(MAIN_MENU_HELP)


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
            '_misc'      : "Miscillaneous",
            'reload'  : {
                'text'   : 'Reload all ActiveFiberCoupling modules (might be broken)',
                'func'   : reload_modules,
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
                'text'   : 'Help with menu or with a function (func=func_name)',
                'func'   : menu_help,
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

                user_input = shlex.split(user_input)    # splits by tokens smartly
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

                if func_key == 'manual':    # manual is device agnostic
                    args_key = args_key[0]

                if func_key == 'help':      # help is a special case, handled separately
                    menu_help(user_input[1], MENU_DICT)
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

                kwargs.update(str_to_dict(user_input))

                # check whether kwargs are correct
                try:
                    inspect.signature(func).bind(*args, **kwargs)
                except TypeError as e:
                    warnings.warn(f"Invalid keyword arguments: {e}")
                    print("Function call aborted.")
                    continue

                kwargs_print = dict_to_str(kwargs)
                print("Interpreted kwargs:\n" + kwargs_print)
                yn_input = input("Is this correct? (y/n): ").strip()
                if yn_input.lower() == 'y':
                    print(f"Calling {func_key} with args {args_key} and kwargs as above.\n")
                    func(*args, **kwargs)
                    continue
                elif yn_input.lower() == 'n':
                    pass
                else:
                    print(f"Input {yn_input} could not be interpreted")

                print("Function call aborted.")

            except Exception as e:
                func_traceback = traceback.format_exc()
                warnings.warn(f"An error was encountered: {e}\nFull traceback below:\n" + func_traceback)


if __name__ == '__main__':
    main()
