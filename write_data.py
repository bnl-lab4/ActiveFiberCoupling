# This file isn't actually used for antyhing currently, it was just a test

import io
import sys

def start_data():
    file = open("run_data.txt", "w")
    #list = []

    output = io.StringIO()

    original_stdout = sys.stdout

    sys.stdout = output


def end_data():
    
    sys.stdout = original_stdout

    text = output.getvalue()

    output_list = text.splitlines()

    file.writelines(output_list)
