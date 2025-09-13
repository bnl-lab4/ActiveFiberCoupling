import manual_keycontrol
import algo_randomsearch
import algo_gridsearch
import algo_hill_climbing
import algo_crossSearch
import algo_one_cross_section
import algo_three_cross_sections
import algo_calculate
import algo_cross_section_search
import algo_continuous_search
import third_algo_one_cross_section_with_plots
import data_algo_calculate
import data_algo_continuous_search
import data_2_algo_calculate
import data_2_algo_continuous_search
import center_axes
import zero_axes
import photodiode_in
import algo_focal_estimator
import algo_PSO


MENU_DICT = {
        '_manual_control' : "Manual Control Options",
        '0' : ('Stage 0', ),
        '1' : ('Stage 1', ),
        'center p0' : ('Center piezo axes of stage 0', ),
        'center p1' : ('Center piezo axes of stage 1', ),
        'center s0' : ('Center stepper axes of stage 0', ),
        'center s1' : ('Center stepper axes of stage 1', ),
        'zero p0' : ('Zero piezo axes of stage 0', ),
        'zero p1' : ('Zero piezo axes of stage 1', ),
        'zero s0' : ('Zero stepper axes of stage 0', ),
        'zero s1' : ('Zero stepper axes of stage 1', ),
            }

def display_menu():
    max_choice_length = max(map(len, list(MENU_DICT.keys())))
    whitespace = max_choice_length + 2
    for key, value in MENU_DICT.items():
        if key.startswith('_'):
            print('\n' + value)
        else:
            print(f"{key}:{' ' * whitespace}{value[0]}")
    return

def receive_input(user_input):
    user_input = user_input.strip()
    if user_input not in MENU_DICT.keys() or user_input.startwith('_'):
        print('Invalid input')
    else:
        MENU_DICT[user_input]()
