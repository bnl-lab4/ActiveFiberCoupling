import inspect
import warnings
from typing import List, Sequence, Optional

from MovementClasses import Distance, StageDevices


def parse_str_values(value):
    # Attempt to convert to appropriate data type
    if value.lower() in ['true', 'false']:
        return value.lower() == 'true'

    if value.startswith('{') and value.endswith('}'):
        # Check if value is dict
        value = value[1:-1]
        list_values = [item.strip() for item in value.split(',')]
        for i, val in enumerate(list_values):
            if val.startswith('D('):
                list_values[i] = list_values[i] + ',' + list_values[i+1]
                del list_values[i+1]
        if any(['=' in elem for elem in list_values]):
            raise ValueError("You must use : between keys and values in a dictionary, not =")
        list_values = [elem.split(':') for elem in list_values]
        value = {key : parse_str_values(value) for key, value in list_values}
        return value

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
        if value.startswith('D('):
            value = value[2:-1]
        else:
            value = value[9:-1]
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


def str_to_dict(tokens: List[str]):
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
            if len(sub_dict_list) == 0:
                sub_dict_print = '{}'
            elif len(sub_dict_list) == 1:
                sub_dict_print = '{ ' + sub_dict_list[0] + ' }'
            else:
                sub_dict_print = '{\n' + '\n'.join(sub_dict_list) + "\n}"
            sub_dict_print = sub_dict_print.replace('\n', '\n' + ' ' * 8)
            sequence_print.append(sub_dict_print)
            continue

        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            sub_seq_list = sequence_to_str(elem, joined=False)
            if len(sub_seq_list) == 0:
                sub_seq_print = '()'
            elif len(sub_seq_list) == 1:
                sub_seq_print = '( ' + sub_seq_list[0] + ' )'
            else:
                sub_seq_print = '(\n' + '\n'.join(sub_seq_list) + "\n)"
            sub_seq_print = sub_seq_print.replace('\n', '\n' + ' ' * 8)

            if isinstance(elem, list):
                sub_seq_print = sub_seq_print.replace('(', '[', 1)
                sub_seq_print = sub_seq_print[::-1].replace(')', ']', 1)

            sequence_print.append(sub_seq_print)
            continue

        elif isinstance(elem, Distance):
            sequence_print.append(elem.prettyprint(stacked=True))
        elif isinstance(elem, StageDevices):
            sequence_print.append(elem.name)
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
            sub_dict_print = f"{str(key)} : "
            if len(sub_dict_list) == 0:
                sub_dict_print += '{}'
            elif len(sub_dict_list) == 1:
                sub_dict_print += '{ ' + sub_dict_list[0] + ' }'
            else:
                sub_dict_print += '{\n' + '\n'.join(sub_dict_list) + "\n}"
            sub_dict_print = sub_dict_print.replace('\n', '\n' + ' ' * 8)
            dict_print.append(sub_dict_print)
            continue

        elif isinstance(value, Sequence) and not isinstance(value, str):
            sub_seq_list = sequence_to_str(value, joined=False)
            sub_seq_print = f"{str(key)} : "
            if len(sub_seq_list) == 0:
                sub_seq_print += '()'
            elif len(sub_seq_list) == 1:
                sub_seq_print += '( ' + sub_seq_list[0] + ' )'
            else:
                sub_seq_print += '(\n' + '\n'.join(sub_seq_list) + "\n)"
            sub_seq_print = sub_seq_print.replace('\n', '\n' + ' ' * 8)

            if isinstance(value, list):
                sub_seq_print = sub_seq_print.replace('(', '[', 1)
                sub_seq_print = sub_seq_print[::-1].replace(')', ']', 1)[::-1]

            dict_print.append(sub_seq_print)
            continue

        elif isinstance(value, Distance):
            dict_print.append(f"{str(key)} : {value.prettyprint(stacked=True)}")
        elif isinstance(value, StageDevices):
            dict_print.append(f"{str(key)} : {value.name}")
        else:
            dict_print.append(f"{str(key)} : {str(value)}")

    if joined:
        dict_print = '\n'.join(dict_print)
    return dict_print


def menu_help(func_key: Optional[str] = None, menu: Optional[dict] = None):
    MAIN_MENU_HELP = """
MAIN MENU HELP
Function call syntax is '<func name> <stagenum device> <default kwarg name> <key=value> <key=value>'.
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
center 0s
        -- Centers the steppers of stage 0.
grid 0s fine axes='yz' planes=[D(100,"fullsteps"),D(500,fullsteps)]
        -- Run grid search on stage 0 with steppers in y-z planes,
            using the 'fine' preset kwargs but overriding the planes argument.
    """

    if func_key is not None and menu is not None:
        func_dict_str = dict_to_str(menu[func_key].to_dict())
        func_sig = inspect.signature(menu[func_key].func)
        siglist = []
        for _, sig in list(func_sig.parameters.items()):
            siglist.append(str(sig))
        sig_str = ',\n'.join(siglist)
        return print(f"FUNCTION MENU ENTRY:\n{func_dict_str}" + '\n'*2 +
                     f"FUNCTION SIGNATURE:\n{sig_str}")

    return print(MAIN_MENU_HELP)
