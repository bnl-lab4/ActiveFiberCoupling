import warnings
import os
import logging
from typing import Optional, Union

SAFE_MODULES = ['MovementClasses', 'MovementUtils', 'StringUtils', 'SensorClasses',
                'SimulationClasses', 'HillClimb', 'grid_search', 'Distance',
                'manual_control', 'StageStatus', 'LoggingUtils']

# Initial logging setup
log = logging.getLogger(__name__)


# Add Trace level to logging
TRACE_LEVEL_NUM = 5
logging.TRACE = TRACE_LEVEL_NUM
logging.addLevelName(TRACE_LEVEL_NUM, 'TRACE')


def trace(self, message, *args, **kwargs):
    if self.isEnabledFor(TRACE_LEVEL_NUM):
        self._log(TRACE_LEVEL_NUM, message, args, **kwargs) # unexpanded args is correct


# Attache to Logger class
logging.Logger.trace = trace

# Make trace available globally
logging.trace = lambda msg, *args, **kwargs: logging.log(TRACE_LEVEL_NUM, msg, *args, **kwargs)


class OnlyAFCDebugs(logging.Filter):
    """
    A filter that blocks DEBUG messages from any logger not in the safe list.
    """
    def __init__(self, name='', safe_modules=None):
        super().__init__(name)
        # Convert to a set for fast lookup
        self.safe_modules = set(safe_modules) if safe_modules else set()

    def filter(self, record):
        # 1. Allow all messages that are NOT DEBUG (i.e., INFO, WARNING, etc.)
        if record.levelno > logging.DEBUG:
            return True

        # 2. For DEBUG messages, only allow them if the logger name is in the safe list
        if record.levelno == logging.DEBUG:
            # record.name is the name of the logger that originated the message
            if record.name in self.safe_modules:
                return True # Pass DEBUG from a safe module
            else:
                return False # Block DEBUG from an unsafe module

        return True # Catch-all for any other levels (like custom TRACE)


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
        level = getattr(logging, log_level.upper(), logging.DEBUG)
    elif log_level in (TRACE_LEVEL_NUM, 10, 20, 30, 40, 50):
        level = log_level
    else:
        raise ValueError(f"log_level {log_level} is not in an acceptable form")

    if console_log_level is None or not log_to_console:
        console_level = logging.INFO    # INFO for now, WARNING during science use
    elif isinstance(console_log_level, str):
        # Map string to logging level
        console_level = getattr(logging, console_log_level.upper(), logging.INFO)
    elif console_log_level in (TRACE_LEVEL_NUM, 10, 20, 30, 40, 50):
        console_level = console_log_level
    else:
        raise ValueError(f"console_log_level {console_log_level} is not in an acceptable form")

    standard_format = logging.Formatter("%(asctime)s-%(levelname)s-" +
                "%(module)s-line%(lineno)d-%(funcName)s :: %(message)s")
    caught_warning_format = logging.Formatter("%(asctime)s-WARNING(CAUGHT)-%(message)s")
    debug_filter = OnlyAFCDebugs(safe_modules=SAFE_MODULES)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    warnings_logger = logging.getLogger('py.warnings')
    warnings_logger.setLevel(logging.WARNING)
    warnings_logger.propagate = False    # so root_logger does not get already caught warning logs

    # clears existing handlers to avoid duplicates
    if root_logger.hasHandlers():
        for handler in root_logger.handlers:
            handler.close()
        root_logger.handlers.clear()
    if warnings_logger.hasHandlers():
        for handler in warnings_logger.handlers:
            handler.close()
        warnings_logger.handlers.clear()

    # logic for which level done above, never truly off
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(standard_format)
    console_handler.addFilter(debug_filter)
    root_logger.addHandler(console_handler)

    console_handler_warnings = logging.StreamHandler()
    console_handler_warnings.setLevel(console_level)
    console_handler_warnings.setFormatter(caught_warning_format)
    warnings_logger.addHandler(console_handler_warnings)

    # Check and add file handling
    if log_to_file and filename:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(standard_format)
        file_handler.addFilter(debug_filter)
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
        user_input = input("Console log level? [trace/debug/info/warning/error/critical/skip]: ").strip().lower()
        if user_input == 's' or user_input == 'skip':
            break
        try:
            LoggingSettings['Console log level'] = getattr(logging, user_input.upper())
            break
        except AttributeError:
            print(f"Could not find log level matching {user_input}")
        print(f"Could not update log level to {user_input}")

    while True: # log level?
        user_input = input("Log level? [trace/debug/info/warning/error/critical/skip]: ").strip().lower()
        if user_input == 's' or user_input == 'skip':
            break
        try:
            LoggingSettings['Log level'] = getattr(logging, user_input.upper())
            break
        except AttributeError:
            print(f"Could not find log level matching {user_input}")
        print(f"Could not update log level to {user_input}")

    setup_logging(*list(LoggingSettings.values()))
