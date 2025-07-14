import serial
from motion import move
import photodiode_in

def run(ser):
    print("Manual control (typed input) mode. Type 'exit' to return to menu.")
    while True:
        user_input = input("Enter command (e.g., x 1.2): ").strip()
        if user_input.lower() == 'exit':
            break
        try:
            axis, value = user_input.split()
            move(axis, float(value), ser)
            #photodiode_in.print_avg_stdv()
        except Exception as e:
            print(f"Invalid input: {e}")
