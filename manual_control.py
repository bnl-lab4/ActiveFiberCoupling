import serial
from motion import move

def run(ser):
    print("Manual control (typed input) mode. Type 'exit' to return to menu.")
    while True:
        user_input = input("Enter command (e.g., x 1.2): ").strip()
        if user_input.lower() == 'exit':
            break
        try:
            axis, value = user_input.split()
            move(axis, float(value), ser)
        except Exception as e:
            print(f"Invalid input: {e}")
