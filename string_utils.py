"""
Utility functions for parsing strings to dictionaries and back, and the
help function for the main menu in `main.py`.
"""

import inspect
import warnings
from collections.abc import Callable
from typing import List, Optional, Sequence, Union

from logging_utils import get_logger
from movement_classes import Distance, StageDevices

# Initial logging
logger = get_logger(__name__)


def parse_str_values(value):
    """
    Parse a string into objects such as dicts, lists, and `Distance.Distance`.

    Parses a string into python objects, defaulting to a string if no other
    parsing is available. Lists must start and end with brackets (``[]``)
    and have comma-separated elements. Dictionaries must start and end with
    braces (``{}``) and have comma-separated key-value pairs which are
    separated by a colon (``:``). `Distance.Distance` objects must start
    with ``Distance(`` or ``D(`` and end with a closing parenthesis. If a
    unit is provided, it must be separated from the value by a comma. The
    keywords 'true', 'false', and 'none' (case-insensitive) are parsed as
    the python objects ``True``, ``False``, and ``None``, respectively.

    Parameters
    ----------
    value : str
        The string to be parsed.

    Returns
    -------
    str or dict or list `Distance.Distance` or bool or None
        Returns the python object `value` was parsed to be.

    Notes
    -----
    This function calls itself recursively to parse the elements of dicts,
    lists, and `Distance.Distance` objects.
    """
    # Check if value is a keyword
    if value.lower() in ["true", "false"]:
        return value.lower() == "true"

    if value.lower() == "none":
        return None

    # Check if value is dict
    if value.startswith("{") and value.endswith("}"):
        value = value[1:-1]
        list_values = [item.strip() for item in value.split(",")]
        for i, val in enumerate(list_values):
            # clean up Distance entries if needed
            if (
                val.startswith("D(") or val.startswith("Distance(")
            ) and not val.endswith(")"):
                list_values[i] = list_values[i] + "," + list_values[i + 1]
                del list_values[i + 1]
        if any(["=" in elem for elem in list_values]):
            raise ValueError(
                "You must use : between keys and values in a dictionary, not ="
            )
        list_values = [elem.split(":") for elem in list_values]
        value = {key: parse_str_values(value) for key, value in list_values}
        return value

    # Check if value is list
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
        list_values = [item.strip() for item in value.split(",")]
        for i, val in enumerate(list_values):
            # clean up Distance entries if needed
            if (
                val.startswith("D(") or val.startswith("Distance(")
            ) and not val.endswith(")"):
                list_values[i] = list_values[i] + "," + list_values[i + 1]
                del list_values[i + 1]
        value = [parse_str_values(value) for value in list_values]
        return value

    # Check if value is meant to be a Distance object
    if (value.startswith("D(") or value.startswith("Distance(")) and value.endswith(
        ")"
    ):
        if value.startswith("D("):
            value = value[2:-1]
        else:
            value = value[9:-1]
        value = [item.strip() for item in value.split(",")]
        return Distance(*value)

    # Check for float or integer
    if value.replace(".", "", 1).isdigit():
        if "." in value:
            return float(value)
        return int(value)

    # Check for quotes indicating a string
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]

    # Return original value if it cannot be parsed (or is just a string)
    return value


def str_to_dict(tokens: List[str]):
    """
    Parses a list of strings into the key-value pairs of a dictionary.
    See `parse_str_values`.
    """
    if isinstance(tokens, str):
        tokens = [
            tokens,
        ]
    assert isinstance(tokens, list), "tokens must be a string or list thereof"

    kwargs_dict = {}
    rejects = []
    for pair in tokens:
        if "=" in pair:  # require '=' to separate key and value
            key, value = [part.strip() for part in pair.split("=", 1)]
            value = parse_str_values(value)
            kwargs_dict[key] = value
        else:
            rejects.append(pair)
    if len(rejects) != 0:
        warnings.warn(
            "The following input kwargs could not be parsed:" + "\n".join(rejects)
        )

    return kwargs_dict


def sequence_to_str(sequence, joined=True) -> Union[str, List[str]]:
    """
    Turns a sequence into formatted a string for printing. See `dict_to_str`.
    """
    sequence_print = []
    for elem in sequence:
        if isinstance(elem, dict):
            sub_dict_list = dict_to_str(elem, joined=False)
            if len(sub_dict_list) == 0:
                sub_dict_print = "{}"
            elif len(sub_dict_list) == 1:
                sub_dict_print = "{ " + sub_dict_list[0] + " }"
            else:
                sub_dict_print = "{\n" + "\n".join(sub_dict_list) + "\n}"
            sub_dict_print = sub_dict_print.replace("\n", "\n" + " " * 8)
            sequence_print.append(sub_dict_print)
            continue

        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            sub_seq_list = sequence_to_str(elem, joined=False)
            if len(sub_seq_list) == 0:
                sub_seq_print = "()"
            elif len(sub_seq_list) == 1:
                sub_seq_print = "( " + sub_seq_list[0] + " )"
            else:
                sub_seq_print = "(\n" + "\n".join(sub_seq_list) + "\n)"
            sub_seq_print = sub_seq_print.replace("\n", "\n" + " " * 8)

            if isinstance(elem, list):
                sub_seq_print = sub_seq_print.replace("(", "[", 1)
                sub_seq_print = sub_seq_print[::-1].replace(")", "]", 1)

            sequence_print.append(sub_seq_print)
            continue

        elif isinstance(elem, Distance):
            sequence_print.append(elem.prettyprint(stacked=True))
        elif isinstance(elem, StageDevices):
            sequence_print.append(elem.name)
        elif isinstance(elem, Callable):
            sequence_print.append(f"{elem.__module__}.{elem.__name__}()")
        else:
            sequence_print.append(str(elem))

    if joined:
        sequence_print = "\n".join(sequence_print)
    return sequence_print


def dict_to_str(mydict, joined=True) -> Union[str, List[str]]:
    """
    Turns a dict into a formatted string for printing.

    Takes a dictionary with keys and values of arbitrary type and outputs
    a string showing the contents. `str()` is called on the key. If the
    value is a dict or a sequence, this function calls itself or
    `sequence_to_str`, respectively. If the value is a `Distance.Distance`,
    then its ``prettyprint`` method is called. If the value is a
    `movement_classes.StageDevices` object, the ``.name`` attribute is used.
    If the value is a callable, then the value's module-name and name are
    output. Otherwise, `str()` is called on the value.

    Parameters
    ----------
    mydict : dict
        The dictionary to be turned into a string.
    joined : bool, default=True
        Whether to return the dictionary as one string with the key-value
        pairs separated by a ``\n`` (default) or as a list of key-value
        strings.

    Returns
    -------
    str or List[str]
        Single string (``joined=True``) or list of strings
        (``joined=False``) representing `mydict`.

    Notes
    -----
    This function calls itself recursively if a value is a dict.
    """
    dict_print = []
    for key, value in mydict.items():
        if isinstance(value, dict):
            sub_dict_list = dict_to_str(value, joined=False)
            sub_dict_print = f"{str(key)} : "
            if len(sub_dict_list) == 0:
                sub_dict_print += "{}"
            elif len(sub_dict_list) == 1:
                sub_dict_print += "{ " + sub_dict_list[0] + " }"
            else:
                sub_dict_print += "{\n" + "\n".join(sub_dict_list) + "\n}"
            sub_dict_print = sub_dict_print.replace("\n", "\n" + " " * 8)
            dict_print.append(sub_dict_print)
            continue

        elif isinstance(value, Sequence) and not isinstance(value, str):
            sub_seq_list = sequence_to_str(value, joined=False)
            sub_seq_print = f"{str(key)} : "
            if len(sub_seq_list) == 0:
                sub_seq_print += "()"
            elif len(sub_seq_list) == 1:
                sub_seq_print += "( " + sub_seq_list[0] + " )"
            else:
                sub_seq_print += "(\n" + "\n".join(sub_seq_list) + "\n)"
            sub_seq_print = sub_seq_print.replace("\n", "\n" + " " * 8)

            if isinstance(value, list):
                sub_seq_print = sub_seq_print.replace("(", "[", 1)
                sub_seq_print = sub_seq_print[::-1].replace(")", "]", 1)[::-1]

            dict_print.append(sub_seq_print)
            continue

        elif isinstance(value, Distance):
            dict_print.append(f"{str(key)} : {value.prettyprint(stacked=True)}")
        elif isinstance(value, StageDevices):
            dict_print.append(f"{str(key)} : {value.name}")
        elif isinstance(value, Callable):
            dict_print.append(f"{str(key)} : {value.__module__}.{value.__name__}()")
        else:
            dict_print.append(f"{str(key)} : {str(value)}")

    if joined:
        dict_print = "\n".join(dict_print)
    return dict_print


def menu_help(func_key: Optional[str] = None, menu: Optional[dict] = None):
    """
    Prints a help message for the main menu or specific functions in `main.py`.

    Help function for the main menu in `main.py`. Returns a generic help
    message about command input if no function is supplied. If a function
    name is supplied, then the function's entry in the main menu dict and
    the function's signature (args) are returned.

    Parameters
    ----------
    func_key : str, optional
        The name of the main menu option to give a help message for.
    menu : dict, optional
        Menu dictionary from `main.py`.

    Returns
    -------
    str
        String with help message.
    """
    MAIN_MENU_HELP = """
MAIN MENU HELP
Function call syntax is '<func name> <stagenum> < device> <default kwarg name> <key=value> <key=value> ...'.
Call 'help <func name>' to see the required args and optional kwargs.
Space is the delimiter between arguments, so do not use spaces anywhere else.
Keyword argument values can be ints, floats, strings, Distance objects, and lists thereof.
Strings are handled lazily and do not need to be wrapped with ' or " (but can be).
Lists are denoted by starting and ending with brackets '[' ']', with the elements comma separated.
Distance objects are denoted by starting with 'D(' or 'Distance(' and ending with parentheses ')'.
The first argument of Distance is the value, the second is the units, separated by only a comma.

Some examples:
help reload
        -- Show the input parameters of reload.
log
        -- Certain functions don't need an arg.
center 0 s
        -- Centers the steppers of stage 0.
grid 1 s fine axes='yz' planes=[D(100,"fullsteps"),D(500,microns)]
        -- Run grid search on stage 1 with steppers in y-z planes,
            using the 'fine' preset kwargs but overriding the planes argument.
    """

    if func_key is not None and menu is not None:
        func_dict_str = dict_to_str(menu[func_key].to_dict())
        func_sig = inspect.signature(menu[func_key].func)
        siglist = []
        for _, sig in list(func_sig.parameters.items()):
            siglist.append(str(sig))
        sig_str = ",\n".join(siglist)
        return print(
            f"FUNCTION MENU ENTRY:\n{func_dict_str}"
            + "\n" * 2
            + f"FUNCTION SIGNATURE:\n{sig_str}"
        )

    return print(MAIN_MENU_HELP)
