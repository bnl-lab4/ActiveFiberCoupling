################ TODO
# recursive hill climbing
# deviation angles in 3d fit
# parse unit abbreviations in kwargs
# populate default kwargs dictionaries
# useful exceptions for input parsing
# better handling of non-connected stage devices
# rewrite command line arg handling to be more flexible (dict?)
#####################


from __future__ import annotations  # for type hinting types not yet defined
import os
import sys
import importlib  # for reloading modules
import logging
import warnings
import contextlib  # better than nested 'with' statements
import readline  # enables input() to save history just by being loaded
import atexit  # for saving readline history file
import inspect  # checking input kwargs match to function in menu
import traceback  # show traceback in main menu
from typing import Tuple, Dict, Optional, List, Union, Any, cast
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
from LoggingUtils import get_logger
import ContinuousReadout
from hardware_interfaces import TicUSB, Serial, DAQ, HardwareLibMissingStub


# Device info constants
SOCKET0 = dict(host="192.168.0.100", port=8000, sensortype=SensorType.SOCKET)
SOCKET1 = dict(host="192.168.0.100", port=8000, sensortype=SensorType.SOCKET)
PIPLATE0 = dict(addr=0, channel=0, sensortype=SensorType.PIPLATE)
PIPLATE1 = dict(addr=0, channel=1, sensortype=SensorType.PIPLATE)
SIMSENSOR_REPLACEMENT = dict(
    propagation_axis="y", focal_ratio=4.0, angle_of_deviation=0
)
SIMSENSOR_ASPH = dict(propagation_axis="y", focal_ratio=4.0, angle_of_deviation=3 / 180)
SIMSENSOR_LABTELE = dict(propagation_axis="y", focal_ratio=28.0, angle_of_deviation=0)
SIMSENSOR_SKYTELE = dict(propagation_axis="y", focal_ratio=7.0, angle_of_deviation=0)

STEPPER_DICT0 = dict(x="00485175", y="00485185", z="00485159")
STEPPER_DICT1 = dict(x="00485149", y="00485151", z="00485168")

STAGENAME_LIST: Tuple[str, ...] = (
    "stage0",
    "stage1",
    "simstage_asph",
    "simstage_labtel",
    "simstage_skytel",
)
SENSOR_LIST: Tuple[Dict[str, Any], ...] = (
    PIPLATE0,
    PIPLATE1,
    SIMSENSOR_ASPH,
    SIMSENSOR_LABTELE,
    SIMSENSOR_SKYTELE,
)
PIEZO_PORT_LIST: Tuple[Optional[str], ...] = (
    "/dev/ttyACM0",
    "/dev/ttyACM1",
    None,
    None,
    None,
)
STEPPER_DICT_LIST: Tuple[Optional[Dict[str, str]], ...] = (
    STEPPER_DICT0,
    STEPPER_DICT1,
    None,
    None,
    None,
)


MOVEMENT_TYPE_MAP = {
    "p": MovementType.PIEZO,
    "s": MovementType.STEPPER,
    "g": MovementType.GENERAL,
}

WHICH_DEVICE_MAP = {"p": "piezo", "s": "stepper", "r": "sensor"}

COMMAND_ARG_VALUE_DICT = dict(t=True, y=True, true=True, f=False, n=False, false=False)


# custom warning format to remove the source line
def custom_formatwarning(message, category, filename, lineno, _=None):
    # Custom warning formatter that removes the source line.
    return f"{filename}-line{lineno}-{category.__name__} :: {message}\n"


# Initial logging/warning configuration
logger = get_logger(__name__)


def accept_input_args(
    input_tuple: List[Union[bool, str, int, None]], input_args: List[str]
) -> list:
    # relies upon order of input_tuple, should be refactored

    for arg in input_args:
        if "-" not in arg:
            warnings.warn(
                f"Command-line argument {arg} not formatted properly (arg-bool)"
            )
            continue
        arg, val = arg.strip().lower().split("-")
        if arg == "autohome":
            input_tuple[0] = COMMAND_ARG_VALUE_DICT[val]
            continue
        if arg == "requireconnection" or arg == "requirecon":
            input_tuple[1] = COMMAND_ARG_VALUE_DICT[val]
            continue
        if arg == "logtoconsole":
            input_tuple[2] = COMMAND_ARG_VALUE_DICT[val]
            continue
        if arg == "logtofile":
            input_tuple[3] = COMMAND_ARG_VALUE_DICT[val]
            continue
        if arg == "logfile":
            if LoggingUtils.verify_logfile(val):
                input_tuple[4] = val
        if arg == "consoleloglevel":  # not working? (at least with trace)
            try:
                input_tuple[5] = getattr(logging, val)
            except AttributeError:
                warnings.warn(f"Could not find log level {val}, default will be kept")
                continue
            input_tuple[5] = val
            continue
        if arg == "loglevel":
            try:
                input_tuple[6] = getattr(logging, val)
            except AttributeError:
                warnings.warn(f"Could not find log level {val}, default will be kept")
                continue
            input_tuple[6] = val
            continue

        warnings.warn(f"{'-'.join(arg)} is not a valid argument")

    return input_tuple


def update_exposure_time(texp: float) -> float:
    print(
        f"Update default exposure time (currently {texp}):"
        + "(iterations for piplates, milliseconds for sockets)"
    )
    user_input = input(">> ").strip()
    try:
        user_input = int(float(user_input))
    except ValueError:
        print(
            f"Input {user_input} cannot be converted to an int. texp will remain at {texp}"
        )
        return texp
    texp = user_input
    return texp


class MenuEntry:
    def __init__(
        self,
        text: str,
        func: Optional[Callable] = None,
        args_config: Tuple[str, ...] = (),
        kwargs_config: Dict["str", dict] = {},
    ):
        self.text = text
        self.func = func
        self.args_config = args_config
        self.kwargs_config = kwargs_config

    def to_dict(self):
        return dict(
            text=self.text,
            func=self.func,
            args=self.args_config,
            kwargs=self.kwargs_config,
        )

    def execute(
        self, controller: ProgramController, user_input_parts: List[str]
    ) -> None:
        # Handle special case for help function
        if self.func is None:
            logger.warning("No function assigned to this menu entry")
            return
        if self.func == StringUtils.menu_help:
            target_func = user_input_parts[0] if user_input_parts else None
            self.func(target_func, controller.menu)
            return

        resolved_args = []
        if "stage" in self.args_config:
            try:
                resolved_args.append(controller.stages[int(user_input_parts[0])])
            except KeyError as e:
                warnings.warn(f"{e} is not an acceptable stage key")
                print("Function call aborted")
                return
        if "MovementType" in self.args_config:
            try:
                resolved_args.append(MOVEMENT_TYPE_MAP[user_input_parts[1]])
            except KeyError as e:
                warnings.warn(f"{e} is not an acceptable movement type key")
                print("Function call aborted")
                return
        if "exposure_time" in self.args_config:
            resolved_args.append(controller.exposure_time)

        args_len = (
            len(self.args_config) - 1
            if "exposure_time" in self.args_config
            else len(self.args_config)
        )
        kwargs = {}
        if len(user_input_parts) > args_len:
            if "=" not in user_input_parts[args_len]:
                try:
                    kwargs_default_key = user_input_parts[args_len].lower()
                    kwargs.update(self.kwargs_config[kwargs_default_key])
                    args_len += 1
                except KeyError as e:
                    warnings.warn(
                        f"{e} is not a valid kwarg preset."
                        + " Perhaps you are missing a '=' or added a space?"
                    )
                    print("Function call aborted")
                    return

        if len(user_input_parts) > args_len:
            if any("=" not in kwarg for kwarg in user_input_parts[args_len:]):
                warnings.warn(
                    "One or more kwargs are not in the form '<key>=<value>'."
                    + " Only args and kwargs should be separated by spaces."
                )
                print("Function call aborted")
                return
            kwargs.update(StringUtils.str_to_dict(user_input_parts[args_len:]))

        # if exposure time is entered as a kwarg, replace the arg value
        if "exposure_time" in kwargs.keys() and "exposure_time" in self.args_config:
            resolved_args[-1] = kwargs.pop("exposure_time")

        # check whether kwargs are valid
        try:
            inspect.signature(self.func).bind(*resolved_args, **kwargs)
        except TypeError as e:
            warnings.warn(f"Invalid keyword arguments: {e}")
            print("Function call aborted")
            return

        if kwargs:
            kwargs_print = StringUtils.dict_to_str(kwargs)
            assert isinstance(kwargs_print, str)
            print("Interpreted kwargs:\n" + kwargs_print)
            yn_input = input("Is this correct? (y/n): ").strip().lower()
            print("")
            if yn_input != "y":
                print("Function call aborted")
                return

        args_string = f" with args {resolved_args}" if resolved_args else ""
        kwargs_string = (
            f" with kwargs {StringUtils.dict_to_str(kwargs)}" if kwargs else ""
        )
        if args_string and kwargs_string:
            args_string += " and"
        logger.debug(f"Calling {self.func.__name__}" + args_string + kwargs_string)
        self.func(*resolved_args, **kwargs)


class ProgramController:
    def __init__(
        self, autohome: bool, require_connection: bool, logging_settings: tuple
    ):
        self.autohome = autohome
        self.require_connection = require_connection
        self.logging_settings = logging_settings

        self.running = True
        self.exposure_time = 1000  # default value
        self.stages = {}
        self.stack = contextlib.ExitStack()
        self.menu = self._build_menu()

    def __enter__(self):
        LoggingUtils.setup_logging(*self.logging_settings)  # type: ignore [reportOptionalIterable]
        self._initialize_devices()
        return self

    def __exit__(self, _, __, ___):
        self.stack.close()
        logger.debug("Device contexts closed cleanly")
        return False

    def _initialize_devices(self):
        for i, (NAME, SENSOR, PIEZO_PORT, STEPPER_DICT) in enumerate(
            zip(STAGENAME_LIST, SENSOR_LIST, PIEZO_PORT_LIST, STEPPER_DICT_LIST)
        ):
            if "propagation_axis" not in SENSOR.keys():
                if any(
                    [
                        type(lib) is HardwareLibMissingStub
                        for lib in (TicUSB, Serial, DAQ)
                    ]
                ):
                    logger.warning(
                        f"One or more of TicUSB, Serial, DAQ is a stub. Loading replacement simulation classes for stage {NAME}."
                    )
                    sensor_dict = cast(Dict[str, Any], SIMSENSOR_REPLACEMENT)
                    sensor = self.stack.enter_context(SimulationSensor(**sensor_dict))
                    stage = self.stack.enter_context(
                        SimulationStageDevices(NAME + "_SIMSENSOR", sensor=sensor)
                    )
                else:
                    sensor_dict = cast(Dict[str, Any], SENSOR)
                    sensor = self.stack.enter_context(
                        Sensor(sensor_dict, sensor_dict["sensortype"])
                    )
                    stage = self.stack.enter_context(
                        StageDevices(
                            NAME,
                            PIEZO_PORT,
                            STEPPER_DICT,
                            sensor=sensor,
                            require_connection=self.require_connection,
                            autohome=self.autohome,
                        )
                    )
            else:
                sensor_dict = cast(Dict[str, Any], SENSOR)
                sensor = self.stack.enter_context(SimulationSensor(**sensor_dict))
                stage = self.stack.enter_context(
                    SimulationStageDevices(NAME, sensor=sensor)
                )
            self.stages[i] = stage

    def _build_menu(self):
        return {
            "_stage_control": "Stage Control Options",
            "manual": MenuEntry(
                text="Stage manual control",
                func=manual_control.run,
                args_config=("stage", "exposure_time"),
            ),
            "zero": MenuEntry(
                text="Zero piezos or steppers positions",
                func=MovementUtils.zero,
                args_config=("stage", "MovementType"),
            ),
            "center": MenuEntry(
                text="Center piezos or steppers positions",
                func=MovementUtils.center,
                args_config=("stage", "MovementType"),
            ),
            "max": MenuEntry(
                text="Maximize piezos or steppers positions",
                func=MovementUtils.max,
                args_config=("stage", "MovementType"),
            ),
            "energize": MenuEntry(
                text="Energize steppers",
                func=MovementUtils.energize,
                args_config=("stage",),
            ),
            "deenergize": MenuEntry(
                text="Deenergize steppers",
                func=MovementUtils.deenergize,
                args_config=("stage",),
            ),
            "home": MenuEntry(
                text="Home steppers", func=MovementUtils.home, args_config=("stage",)
            ),
            "read": MenuEntry(
                text="Continuous readout of sensor in terminal",
                func=ContinuousReadout.run,
                args_config=("stage", "exposure_time"),
            ),
            "_optimization": "Optimization Algorithms",
            "grid": MenuEntry(
                text="Grid search in one or more planes",
                func=grid_search.run,
                args_config=("stage", "MovementType", "exposure_time"),
            ),
            "hillclimb": MenuEntry(
                text="Hill climbing on one or more axes",
                func=HillClimb.run,
                args_config=("stage", "MovementType", "exposure_time"),
            ),
            "_misc": "Miscellaneous",
            "status": MenuEntry(
                text="Report the status of all or part of a stage",
                func=StageStatus.run,
                args_config=("stage", "exposure_time"),
                kwargs_config={"quick": dict(verbose=False), "log": dict(log=True)},
            ),
            "reload": MenuEntry(text="Reload all ActiveFiberCoupling modules"),
            # reload is a special case, whose function must not be in the menu
            "texp": MenuEntry(
                text="Change the default exposure time",
                func=update_exposure_time,
            ),
            "log": MenuEntry(
                text="Change the logging settings",
                func=LoggingUtils.update_logging,
            ),
            "help": MenuEntry(
                text="Help with the menu or a function ('help <func_name>')",
                func=StringUtils.menu_help,
            ),
        }

    def _display_menu(self):
        # excepting section titles
        max_choice_length = max(
            map(
                lambda s: len(s) if not s.startswith("_") else 0, list(self.menu.keys())
            )
        )  # sorry
        whitespace = max_choice_length + 2  # for aligning descriptions
        for key, value in self.menu.items():
            if key.startswith("_"):
                print("\n" + value)
            else:
                print(f"{key}:{' ' * (whitespace - len(key))}{value.text}")
        return

    def _reload_menu(self):
        for key, entry in self.menu.items():
            if key.startswith("_") or key == "reload":
                continue  # ignore menu section headers and this functions menu entry

            # get name of module function comes from
            func_module_name = entry.func.__module__
            if func_module_name == "__main__":
                continue  # can't reload this script itself

            try:
                func_name = entry.func.__name__
                reloaded_module = importlib.reload(
                    importlib.import_module(func_module_name)
                )
                new_func = getattr(reloaded_module, func_name)
                entry.func = new_func
                logger.info(f"{key} function reloaded succesfully")

            except (ImportError, AttributeError):
                logger.warning(f"{key} function was not reloaded succesfully")

        self.menu = self._build_menu()

    def run(self):
        while self.running:
            try:
                print("\n" * 4)
                self._display_menu()
                print("")
                user_input = input(">> ").strip()
                print("")

                if not user_input:
                    continue

                ui_lower = user_input.lower()
                # special cases
                if ui_lower == "q":
                    self.running = False
                if ui_lower == "exec":  # hidden mode :), pls don't abuse
                    print(
                        "WARNING: IN EXEC MODE\nAnything typed here will be executed as python code."
                    )
                    exec(input("exec >> "))
                    continue
                if ui_lower == "reload":
                    self._reload_menu()
                    continue
                if ui_lower == "uwu":
                    print("UwU")
                    continue

                parts = user_input.split(" ")
                parts = [part for part in parts if part != ""]

                entry_key = parts[0].lower()
                if entry_key not in self.menu.keys() or entry_key.startswith("_"):
                    if entry_key != "q":
                        print("\nInvalid menu option")
                    continue

                # These functions update the state of ProgramController
                if entry_key == "reload":
                    self._reload_menu()
                elif entry_key == "log":
                    self.logging_settings = LoggingUtils.update_logging()
                elif entry_key == "texp":
                    self.exposure_time = update_exposure_time(self.exposure_time)
                else:
                    entry = self.menu[entry_key]
                    entry.execute(self, parts[1:])

            except Exception as e:
                func_traceback = traceback.format_exc()
                warnings.warn(
                    f"An error was encountered: {e}\nFull traceback below:\n{func_traceback}"
                )
            except KeyboardInterrupt:
                logging.warning("KeyboardInterrupt caught")
                print("KeyboardInterrupt was caught, enter 'q' to quit")


def main():
    # warning configuration
    logging.captureWarnings(True)
    warnings.formatwarning = custom_formatwarning

    # load input history file
    history_file = "./.main_history"
    if os.path.exists(history_file):
        readline.read_history_file(history_file)
    readline.set_history_length(500)  # only save the last 500 entries
    # before stopping the script, save the input history to the file
    atexit.register(readline.write_history_file, history_file)

    # Default runtime variable
    autohome = True  # home motors upon establishing connection
    require_connection = (
        False  # raise exception if device connections fail to establish
    )
    logging_settings: List[Any] = [None] * 5

    if len(sys.argv) > 1:
        InputArgs = sys.argv[1:]
        autohome, require_connection, *logging_settings = accept_input_args(
            [autohome, require_connection] + logging_settings, InputArgs
        )

    try:
        with ProgramController(
            autohome, require_connection, tuple(logging_settings)
        ) as controller:
            controller.run()
    except Exception as e:
        logger.critical(f"A critical error occurred during initialization: {e}")
        logger.critical(traceback.format_exc())
        raise e


if __name__ == "__main__":
    main()
