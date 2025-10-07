################ TODO
# CONTINUOUS SENSOR READOUT FUNCTON
# populate default kwargs dictionaries
# add general movement mode to manual control
# rewrite command line arg handling to be more flexible (dict?)
#####################


import os
import sys
import importlib    # for reloading modules
import logging
import warnings
import contextlib   # better than nested 'with' statements
import readline     # enables input() to save history just by being loaded
import atexit       # for saving readline history file
import inspect      # checking input kwargs match to function in menu
import traceback    # show traceback in main menu
from typing import Tuple, Dict, Optional
from collections.abc import Callable

# Import alignment algorithms and control modes
from MovementClasses import StageDevices, MovementType
from SensorClasses import Sensor, SensorType
from SimulationClasses import SimulationSensor, SimulationStageDevices
import manual_control
import MovementUtils
import grid_search
import StringUtils
import HillClimb
import StageStatus
import LoggingUtils
import ContinuousReadout


# Device info constants
SOCKET0 = dict(host = '192.168.1.10', port = 8000, sensortype = SensorType.SOCKET)
SIPM0 = dict(addr = 0, channel = 1, sensortype = SensorType.SIPM)
SIPM1 = dict(addr = 0, channel = 2, sensortype = SensorType.SIPM)
PHOTODIODE0 = dict(addr = 0, channel = 1, sensortype = SensorType.PHOTODIODE)
PHOTODIODE1 = dict(addr = 0, channel = 2, sensortype = SensorType.PHOTODIODE)
SIMSENSOR_ASPH = dict(propagation_axis = 'y', focal_ratio = 4.0, angle_of_deviation = 0)
SIMSENSOR_LABTELE = dict(propagation_axis = 'y', focal_ratio = 28.0, angle_of_deviation = 0)
SIMSENSOR_SKYTELE = dict(propagation_axis = 'y', focal_ratio = 7.0, angle_of_deviation = 0)

SENSOR0 = SIPM0
SENSOR1 = SIMSENSOR_ASPH

PIEZO_PORT0 = '/dev/ttyACM0'
PIEZO_PORT1 = '/dev/ttyACM1'
BAUD_RATE = 115200

STEPPER_DICT0 = dict(x = '00485175', y = '00485185', z = '00485159')
STEPPER_DICT1 = dict(x = None, y = None, z = None)

MOVEMENT_TYPE_MAP = {
    'p': MovementType.PIEZO,
    's': MovementType.STEPPER,
    'g': MovementType.GENERAL
}

WHICH_DEVICE_MAP = {
    'p': 'piezo',
    's': 'stepper',
    'r': 'sensor'
}


# custom warning format to remove the source line
def custom_formatwarning(message, category, filename, lineno, line=None):
    # Custom warning formatter that removes the source line.
    return f"{filename}-line{lineno}-{category.__name__} :: {message}\n"


# Initial logging/warning configuration
log = logging.getLogger(__name__)


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
            if LoggingUtils.verify_logfile(val):
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


def Update_ExposureTime(texp):
    print(f"Update default exposure time (currently {texp}):" +
        "(iterations for SiPMs or photodiodes, milliseconds for sockets)")
    user_input = input(">> ").strip()
    try:
        user_input = int(float(user_input))
    except ValueError:
        print(f"Input {user_input} cannot be converted to an int. texp will remain at {texp}")
        return False
    texp = user_input
    return texp


class MenuEntry:
    def __init__(self, text: str, func: Optional[Callable] = None, args_config: Tuple[str, ...] = (),
                 kwargs_config: Dict['str', dict] = {}):
        self.text = text
        self.func = func
        self.args_config = args_config
        self.kwargs_config = kwargs_config

    def to_dict(self):
        return dict(text=self.text, func=self.func, args=self.args_config,
                    kwargs=self.kwargs_config)

    def execute(self, controller, user_input_parts):
        # Handle special case for help function
        if self.func == StringUtils.menu_help:
            target_func = user_input_parts[0] if user_input_parts else None
            self.func(target_func, controller.menu)
            return

        resolved_args = []
        if 'stage' in self.args_config:
            resolved_args.append(controller.stages[user_input_parts[0]])
        if 'MovementType' in self.args_config:
            resolved_args.append(MOVEMENT_TYPE_MAP[user_input_parts[1]])
        if 'ExposureTime' in self.args_config:
            resolved_args.append(controller.exposure_time)

        args_len = len(self.args_config) - 1 if 'ExposureTime' in self.args_config else len(self.args_config)
        kwargs = {}
        if len(user_input_parts) > args_len:
            if '=' not in user_input_parts[args_len]:
                kwargs_default_key = user_input_parts[args_len].lower()
                kwargs.update(self.kwargs_config[kwargs_default_key])
                args_len += 1

        if len(user_input_parts) > args_len:
            if any('=' not in kwarg for kwarg in user_input_parts[args_len:]):
                warnings.warn("All kwargs must be in <key>=<value> form")
                print("Function call aborted")
                return
            kwargs.update(StringUtils.str_to_dict(user_input_parts[args_len:]))

        # check whether kwargs are valid
        try:
            inspect.signature(self.func).bind(*resolved_args, **kwargs)
        except TypeError as e:
            warnings.warn(f"Invalid keyword arguments: {e}")
            print("Function call aborted")
            return

        if kwargs:
            kwargs_print = StringUtils.dict_to_str(kwargs)
            print("Interpreted kwargs:\n" + kwargs_print)
            yn_input = input("Is this correct? (y/n): ").strip().lower()
            print('')
            if yn_input != 'y':
                print("Function call aborted")
                return

        args_string = f" with args {resolved_args}" if resolved_args else ''
        kwargs_string = f" with kwargs {StringUtils.dict_to_str(kwargs)}" if kwargs else ''
        if args_string and kwargs_string:
            args_string += ' and'
        log.debug(f"Calling {self.func.__name__}" + args_string + kwargs_string)
        self.func(*resolved_args, **kwargs)


class ProgramController:
    def __init__(self, autohome: bool, require_connection: bool, logging_settings: tuple):
        self.autohome = autohome
        self.require_connection = require_connection
        self.logging_settings = logging_settings

        self.running = True
        self.exposure_time = 1000   # default value
        self.stages = {}
        self.stack = contextlib.ExitStack()
        self.menu = self._build_menu()

    def __enter__(self):
        LoggingUtils.setup_logging(*self.logging_settings)
        self._initialize_devices()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stack.close()
        log.debug("Device contexts closed cleanly")
        return False

    def _initialize_devices(self):
        if "propogation_axis" not in SENSOR0:
            sensor0 = self.stack.enter_context(Sensor(SENSOR0, SENSOR0['sensortype']))
            stage0 = self.stack.enter_context(StageDevices('stage0', PIEZO_PORT0, STEPPER_DICT0,
                                        sensor = sensor0, require_connection = self.require_connection,
                                                 autohome = self.autohome))
        else:
            sensor0 = self.stack.enter_context(SimulationSensor(**SENSOR0))
            stage0 = self.stack.enter_context(SimulationStageDevices('simstage0', sensor = sensor0))
        self.stages['0'] = stage0

        if "propagation_axis" not in SENSOR1:
            sensor1 = self.stack.enter_context(Sensor(SENSOR1, SENSOR1['sensortype']))
            stage1 = self.stack.enter_context(StageDevices('stage1', PIEZO_PORT1, STEPPER_DICT1,
                                        sensor = sensor1, require_connection = self.require_connection,
                                                 autohome = self.autohome))

        else:
            sensor1 = self.stack.enter_context(SimulationSensor(**SENSOR1))
            stage1 = self.stack.enter_context(SimulationStageDevices('simstage1', sensor = sensor1))
        self.stages['1'] = stage1

    def _build_menu(self):
        return {
                '_stage_control' : 'Stage Control Options',
                'manual' : MenuEntry(
                    text = 'Stage manual control', func = manual_control.run,
                    args_config = ('stage', 'ExposureTime')
                    ),
                'center' : MenuEntry(
                    text = 'Center piezos or steppers', func = MovementUtils.center,
                    args_config = ('stage', 'MovementType')
                    ),
                'zero' : MenuEntry(
                    text = 'Zero piezos or steppers', func = MovementUtils.zero,
                    args_config = ('stage', 'MovementType')
                    ),
                'energize' : MenuEntry(
                    text = 'Energize steppers', func = MovementUtils.energize,
                    args_config = ('stage',)
                    ),
                'deenergize' : MenuEntry(
                    text = 'Deenergize steppers', func = MovementUtils.deenergize,
                    args_config = ('stage', )
                    ),
                'home' : MenuEntry(
                    text = 'Home steppers', func = MovementUtils.home,
                    args_config = ('stage', )
                    ),
                'read' : MenuEntry(
                    text = 'Continuous readout of sensor in terminal',
                    func = ContinuousReadout.run,
                    args_config = ('stage', 'ExposureTime')
                    ),
                '_optimization' : 'Optimization Algorithms',
                'grid' : MenuEntry(
                    text = 'Grid search in one or more planes',
                    func = grid_search.run,
                    args_config = ('stage', 'MovementType', 'ExposureTime')
                    ),
                'hillclimb' : MenuEntry(
                    text = 'Hill climbing on one or more axes', func = HillClimb.run,
                    args_config = ('stage', 'MovementType', 'ExposureTime')
                    ),
                '_misc' : 'Miscellaneous',
                'status' : MenuEntry(
                    text = 'Report the status of all or part of a stage',
                    func = StageStatus.run,
                    args_config = ('stage', 'ExposureTime'),
                    kwargs_config = {
                        'quick' : dict(verbose = False),
                        'log' : dict(log = True)
                        },
                    ),
                'reload' : MenuEntry(text = 'Reload all ActiveFiberCoupling modules'),
                # reload is a special case, whose function must not be in the menu
                'texp' : MenuEntry(
                    text = 'Change the default exposure time', func = Update_ExposureTime,
                    ),
                'log' : MenuEntry(
                    text = 'Change the logging settings',
                    func = LoggingUtils.Update_Logging,
                    ),
                'help' : MenuEntry(
                    text = "Help with the menu or a function ('help <func_name>')",
                    func = StringUtils.menu_help,
                    )
                }

    def _display_menu(self):
        max_choice_length = max(map(lambda s: len(s) if not s.startswith('_') else 0,
                                    list(self.menu.keys())))    # excepting section titles, sorry
        whitespace = max_choice_length + 2    # for aligning descriptions
        for key, value in self.menu.items():
            if key.startswith('_'):
                print('\n' + value)
            else:
                print(f"{key}:{' ' * (whitespace - len(key))}{value.text}")
        return

    def _reload_menu(self):
        for key, entry in self.menu.items():
            if key.startswith('_') or key == 'reload':
                continue    # ignore menu section headers and this functions menu entry

            # get name of module function comes from
            func_module_name = entry.func.__module__
            if func_module_name == '__main__':
                continue    # can't reload this script itself

            try:
                func_name = entry.func.__name__
                reloaded_module = importlib.reload(importlib.import_module(func_module_name))
                new_func = getattr(reloaded_module, func_name)
                entry.func = new_func
                log.info(f"{key} function reloaded succesfully")

            except (ImportError, AttributeError):
                log.warning(f"{key} function was not reloaded succesfully")

        self.menu = self._build_menu()

    def run(self):
        while self.running:
            try:
                print('\n' * 4)
                self._display_menu()
                print('')
                user_input = input(">> ").strip()
                print('')

                if not user_input:
                    continue

                ui_lower = user_input.lower()
                # special cases
                if ui_lower == 'q':
                    self.running = False
                if ui_lower == 'exec':    # hidden mode :), pls don't abuse
                    print("WARNING: IN EXEC MODE\nAnything typed here will be executed as python code.")
                    exec(input('exec >> '))
                    continue
                if ui_lower == 'reload':
                    self._reload_menu()
                    continue
                if ui_lower == 'uwu':
                    print('UwU')
                    continue

                parts = user_input.split(' ')
                entry_key = parts[0].lower()

                if entry_key not in self.menu.keys() or entry_key.startswith('_'):
                    if entry_key != 'q':
                        print('\nInvalid command')
                    continue

                # These functions update the state of ProgramController
                if entry_key == 'reload':
                    self._reload_menu()
                elif entry_key == 'log':
                    self.logging_settings = LoggingUtils.Update_Logging()
                elif entry_key == 'texp':
                    self.exposure_time = Update_ExposureTime(self.exposure_time)
                else:
                    entry = self.menu[entry_key]
                    entry.execute(self, parts[1:])

            except Exception as e:
                func_traceback = traceback.format_exc()
                warnings.warn(f"An error was encountered: {e}\nFull traceback below:\n{func_traceback}")
            except KeyboardInterrupt:
                logging.warning("KeyboardInterrupt caught")
                print("KeyboardInterrupt was caught, enter 'q' to quit")


def main():
    # warning configuration
    logging.captureWarnings(True)
    warnings.formatwarning = custom_formatwarning

    # load input history file
    history_file = './.main_history'
    if os.path.exists(history_file):
        readline.read_history_file(history_file)
    readline.set_history_length(500)        # only save the last 500 entries
    # before stopping the script, save the input history to the file
    atexit.register(readline.write_history_file, history_file)

    # Default runtime variable
    AutoHome = True # home motors upon establishing connection
    RequireConnection = False # raise exception if device connections fail to establish
    LoggingSettings = [None] * 5

    if len(sys.argv) > 1:
        InputArgs = sys.argv[1:]
        AutoHome, RequireConnection, *LoggingSettings = AcceptInputArgs(
                [AutoHome, RequireConnection] + LoggingSettings, InputArgs)
        # print(f"Received input args: {AutoHome}, {RequireConnection}, {LoggingSettings}")

    try:
        with ProgramController(AutoHome, RequireConnection, LoggingSettings) as controller:
            controller.run()
    except Exception as e:
        log.critical(f"A critical error occurred during initialization: {e}")
        log.critical(traceback.format_exc())


if __name__ == '__main__':
    main()
