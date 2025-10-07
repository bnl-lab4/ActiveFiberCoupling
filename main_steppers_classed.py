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

SENSOR0 = SOCKET0
SENSOR1 = SIMSENSOR_LABTELE

PIEZO_PORT0 = '/dev/ttyACM0'
PIEZO_PORT1 = '/dev/ttyACM1'
BAUD_RATE = 115200

STEPPER_DICT0 = dict(x = '00485175', y = '00485185', z = '00485159')
STEPPER_DICT1 = dict(x = None, y = None, z = None)

# Global mapping for movement type arguments
MOVEMENT_TYPE_MAP = {
    'p': MovementType.PIEZO,
    's': MovementType.STEPPER
}

# --- Global Helper Functions ---

log = logging.getLogger(__name__)

def custom_formatwarning(message, category, filename, lineno, line=None):
    # Custom warning formatter that removes the source line.
    return f"{filename}-line{lineno}-{category.__name__} :: {message}\n"

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
        elif arg in ("requireconnection", "requirecon"):
            inputTuple[1] = CommandArg_ValueDict[val]
        elif arg == 'logtoconsole':
            inputTuple[2] = CommandArg_ValueDict[val]
        elif arg == 'logtofile':
            inputTuple[3] = CommandArg_ValueDict[val]
        elif arg == 'logfile':
            if LoggingUtils.verify_logfile(val):
                inputTuple[4] = val
        elif arg == 'consoleloglevel':
            try:
                inputTuple[5] = getattr(logging, val.upper())
            except AttributeError:
                warnings.warn(f"Could not find log level {val}, default will be kept")
        elif arg == 'loglevel':
            try:
                inputTuple[6] = getattr(logging, val.upper())
            except AttributeError:
                warnings.warn(f"Could not find log level {val}, default will be kept")
        else:
            warnings.warn(f"{arg} is not a valid argument")
    return inputTuple

def Update_ExposureTime(texp):
    print(f"Update default exposure time (currently {texp}):" +
        "(iterations for SiPMs or photodiodes, milliseconds for sockets)")
    user_input = input(">> ").strip()
    try:
        new_texp = int(float(user_input))
        return new_texp
    except ValueError:
        print(f"Input {user_input} cannot be converted to an int. texp will remain at {texp}")
        return texp

# --- Object-Oriented Structure ---

class MenuEntry:
    """Encapsulates the data and execution logic for a single menu command."""
    def __init__(self, text, func, arg_map, kwargs_config=None):
        self.text = text
        self.func = func
        self.arg_map = arg_map
        self.kwargs_config = kwargs_config if kwargs_config is not None else {}

    def execute(self, controller, user_input_parts):
        """Resolves arguments and executes the menu function."""
        # Handle special case for help function
        if self.func == StringUtils.menu_help:
            target_func = user_input_parts[0] if user_input_parts else None
            self.func(target_func, controller.menu)
            return

        # Determine args key
        args_key = ''
        if self.arg_map:
            args_key = user_input_parts.pop(0)

        # Allow for sorted keys (e.g., '1s' or 's1')
        sorted_key = ''.join(sorted(args_key))
        if sorted_key in self.arg_map:
             args_key = sorted_key

        arg_descriptors = self.arg_map.get(args_key)
        if arg_descriptors is None:
            print(f"\nInvalid argument specifier: '{args_key}'")
            return

        # Resolve positional arguments
        resolved_args = []
        for desc in arg_descriptors:
            if desc in controller.stages:
                resolved_args.append(controller.stages[desc])
            elif desc in MOVEMENT_TYPE_MAP:
                resolved_args.append(MOVEMENT_TYPE_MAP[desc])
            elif desc == 'ExposureTime':
                resolved_args.append(controller.exposure_time)
            elif desc in StageStatus.run.__code__.co_varnames: # for 'status' args
                resolved_args.append(desc)
            else:
                warnings.warn(f"Could not resolve argument descriptor: {desc}")

        # Resolve keyword arguments
        kwargs = {}
        if user_input_parts and '=' not in user_input_parts[0]:
            kwargs_default_key = user_input_parts.pop(0).lower()
            if kwargs_default_key in self.kwargs_config:
                kwargs.update(self.kwargs_config[kwargs_default_key])
            else:
                warnings.warn(f"Default kwarg key '{kwargs_default_key}' not found.")

        if any(['=' not in kwarg for kwarg in user_input_parts]):
            warnings.warn("All kwargs must be in key=value form. Function call aborted.")
            return

        kwargs.update(StringUtils.str_to_dict(user_input_parts))

        # Validate arguments and confirm execution
        try:
            inspect.signature(self.func).bind(*resolved_args, **kwargs)
        except TypeError as e:
            warnings.warn(f"Invalid keyword arguments: {e}\nFunction call aborted.")
            return

        if kwargs:
            kwargs_print = StringUtils.dict_to_str(kwargs)
            print("Interpreted kwargs:\n" + kwargs_print)
            yn_input = input("Is this correct? (y/n): ").strip().lower()
            if yn_input != 'y':
                print("Function call aborted.")
                return

        print(f"Calling {self.func.__name__} with args for '{args_key}'.\n")
        self.func(*resolved_args, **kwargs)


class ProgramController:
    """Manages application state, devices, and the main execution loop."""
    def __init__(self, autohome, require_connection, logging_settings):
        self.autohome = autohome
        self.require_connection = require_connection
        self.logging_settings = logging_settings

        # State variables
        self.running = True
        self.exposure_time = 5000
        self.stages = {}
        self.stack = contextlib.ExitStack()
        self.menu = self._build_menu()

    def __enter__(self):
        LoggingUtils.setup_logging(*self.logging_settings)
        self._initialize_devices()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stack.close()
        log.info("Device contexts closed cleanly.")

    def _initialize_devices(self):
        """Initializes and connects to all hardware devices using an ExitStack."""
        if "propagation_axis" not in SENSOR0:
            sensor0 = self.stack.enter_context(Sensor(SENSOR0, SENSOR0['sensortype']))
            stage0 = self.stack.enter_context(StageDevices('stage0', PIEZO_PORT0, STEPPER_DICT0,
                                        sensor=sensor0, require_connection=self.require_connection,
                                        autohome=self.autohome))
        else:
            sensor0 = self.stack.enter_context(SimulationSensor(**SENSOR0))
            stage0 = self.stack.enter_context(SimulationStageDevices('simstage0', sensor=sensor0))
        self.stages['0'] = stage0

        if "propagation_axis" not in SENSOR1:
            sensor1 = self.stack.enter_context(Sensor(SENSOR1, SENSOR1['sensortype']))
            stage1 = self.stack.enter_context(StageDevices('stage1', PIEZO_PORT1, STEPPER_DICT1,
                                        sensor=sensor1, require_connection=self.require_connection,
                                        autohome=self.autohome))
        else:
            sensor1 = self.stack.enter_context(SimulationSensor(**SENSOR1))
            stage1 = self.stack.enter_context(SimulationStageDevices('simstage1', sensor=sensor1))
        self.stages['1'] = stage1
        log.info("Devices initialized.")

    def _build_menu(self):
        """Constructs the menu dictionary with MenuEntry objects."""
        return {
            '_stage_control': "Stage Control Options",
            'manual': MenuEntry(
                text='Stage manual control',
                func=manual_control.run,
                arg_map={'0': ['0', 'ExposureTime'], '1': ['1', 'ExposureTime']}
            ),
            'center': MenuEntry(
                text='Center piezo or stepper axes',
                func=MovementUtils.center,
                arg_map={'0p': ['0', 'p'], '1p': ['1', 'p'], '0s': ['0', 's'], '1s': ['1', 's']}
            ),
            'zero': MenuEntry(
                text='Zero piezo or stepper axes',
                func=MovementUtils.zero,
                arg_map={'0p': ['0', 'p'], '1p': ['1', 'p'], '0s': ['0', 's'], '1s': ['1', 's']}
            ),
            'energize': MenuEntry(
                text='Energize steppers',
                func=MovementUtils.energize,
                arg_map={'0': ['0'], '1': ['1']},
                kwargs_config={'all': dict(axes='all')}
            ),
            'deenergize': MenuEntry(
                text='Deenergize steppers',
                func=MovementUtils.deenergize,
                arg_map={'0': ['0'], '1': ['1']},
                kwargs_config={'all': dict(axes='all')}
            ),
            'home': MenuEntry(
                text='Home steppers',
                func=MovementUtils.home,
                arg_map={'0': ['0'], '1': ['1']},
                kwargs_config={'all': dict(axes='all')}
            ),
            'read': MenuEntry(
                text='Continuous readout of sensor in terminal',
                func=ContinuousReadout.run,
                arg_map={'0': ['0', 'ExposureTime'], '1': ['1', 'ExposureTime']},
                kwargs_config={'noavg': dict(avg=False)}
            ),
            '_optimization': 'Optimization Algorithms',
            'grid': MenuEntry(
                text='Grid search with piezos or steppers',
                func=grid_search.run,
                arg_map={
                    '0p': ['0', 'p', 'ExposureTime'], '1p': ['1', 'p', 'ExposureTime'],
                    '0s': ['0', 's', 'ExposureTime'], '1s': ['1', 's', 'ExposureTime'],
                },
                kwargs_config={
                    'coarse': dict(spacing=Distance(15, "volts"), plot=True, planes=3),
                    'fine': dict()
                }
            ),
            'hillclimb': MenuEntry(
                text='Hill climbing with piezos or steppers',
                func=HillClimb.run,
                arg_map={
                    '0p': ['0', 'p', 'ExposureTime'], '0s': ['0', 's', 'ExposureTime'],
                    '1p': ['1', 'p', 'ExposureTime'], '1s': ['1', 's', 'ExposureTime']
                },
            ),
            '_misc': "Miscillaneous",
            'status': MenuEntry(
                text='Report the status of a stage',
                func=StageStatus.run,
                arg_map={
                    '0': ['0', 'ExposureTime', 'all'], '0p': ['0', 'ExposureTime', 'piezo'],
                    '0s': ['0', 'ExposureTime', 'stepper'], '0r': ['0', 'ExposureTime', 'sensor'],
                    '1': ['1', 'ExposureTime', 'all'], '1p': ['1', 'ExposureTime', 'piezo'],
                    '1s': ['1', 'ExposureTime', 'stepper'], '1r': ['1', 'ExposureTime', 'sensor'],
                },
                kwargs_config={'quick': dict(verbose=False), 'log': dict(log=True)}
            ),
            'reload': MenuEntry('Reload all ActiveFiberCoupling modules', self._reload_menu, {}),
            'texp': MenuEntry('Change the default exposure time', Update_ExposureTime, {}),
            'log': MenuEntry('Change the logging settings', LoggingUtils.Update_Logging, {}),
            'help': MenuEntry("Help with menu or with a function ('help <func_name>')", StringUtils.menu_help, {}),
        }

    def _display_menu(self):
        max_len = max(map(lambda s: len(s) if not s.startswith('_') else 0, self.menu.keys()))
        whitespace = max_len + 2
        for key, value in self.menu.items():
            if key.startswith('_'):
                print(f'\n{value}')
            else:
                text = value.text.format(ExposureTime=self.exposure_time)
                print(f"{key}:{' ' * (whitespace - len(key))}{text}")

    def _reload_menu(self):
        """Reloads the modules associated with menu functions."""
        for key, entry in self.menu.items():
            if key.startswith('_') or entry.func is None:
                continue
            
            module_name = entry.func.__module__
            if module_name == '__main__':
                continue # Cannot reload the main script itself
            
            try:
                func_name = entry.func.__name__
                reloaded_module = importlib.reload(importlib.import_module(module_name))
                new_func = getattr(reloaded_module, func_name)
                entry.func = new_func
                log.info(f"'{key}' function reloaded successfully.")
            except (ImportError, AttributeError):
                log.warning(f"'{key}' function was not reloaded successfully.")
        
        # Re-build menu to update method references like self._reload_menu
        self.menu = self._build_menu()

    def run(self):
        """The main application loop for user interaction."""
        while self.running:
            try:
                print('\n' * 4)
                self._display_menu()
                print('')
                user_input = input(">> ").strip()
                print('')

                if not user_input:
                    continue

                # Handle special commands
                cmd_lower = user_input.lower()
                if cmd_lower == 'q':
                    self.running = False
                    continue
                if cmd_lower == 'exec':
                    print("WARNING: IN EXEC MODE\nAnything typed here will be executed.")
                    exec(input('exec >> '))
                    continue
                if cmd_lower == 'uwu':
                    print('UwU')
                    continue
                
                parts = user_input.strip().split(' ')
                func_key = parts[0].lower()

                if func_key not in self.menu or func_key.startswith('_'):
                    print('\nInvalid command')
                    continue

                entry = self.menu[func_key]
                
                # Handle commands that require special state updates or no args
                if func_key in ('reload', 'log'):
                    entry.func()
                elif func_key == 'texp':
                    self.exposure_time = entry.func(self.exposure_time)
                else:
                    entry.execute(self, parts[1:])

            except Exception as e:
                func_traceback = traceback.format_exc()
                warnings.warn(f"An error was encountered: {e}\nFull traceback below:\n{func_traceback}")
            except KeyboardInterrupt:
                logging.warning("KeyboardInterrupt caught. Type 'q' to exit.")

def main():
    # Setup readline history
    history_file = './.main_history'
    if os.path.exists(history_file):
        readline.read_history_file(history_file)
    readline.set_history_length(500)
    atexit.register(readline.write_history_file, history_file)

    # Initial logging/warning configuration
    logging.captureWarnings(True)
    warnings.formatwarning = custom_formatwarning

    # Default runtime variables
    autohome = True
    require_connection = False
    logging_settings = [True, False, None, logging.INFO, logging.DEBUG] # log_to_console, log_to_file, logfile, console_level, file_level

    # Parse command-line arguments
    if len(sys.argv) > 1:
        autohome, require_connection, *logging_settings = AcceptInputArgs(
            [autohome, require_connection] + logging_settings, sys.argv[1:]
        )
    
    try:
        with ProgramController(autohome, require_connection, logging_settings) as controller:
            controller.run()
    except Exception as e:
        log.critical(f"A critical error occurred during initialization: {e}")
        log.critical(traceback.format_exc())

    print("Exiting program.")

if __name__ == '__main__':
    main()
