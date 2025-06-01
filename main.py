#!/usr/bin/python

import manual_control
import algorithm_peaksearch

def main_menu():
    while True:
        print("\nFiber Alignment Control")
        print("1. Manual Control")
        print("2. Peak Search Alignment")
        print("4. Exit")
        choice = input("Select an option: ").strip()

        if choice == '1':
            manual_control.run()
        elif choice == '2':
            algorithm_peaksearch.run()
        elif choice == '4':
            print("Exiting.")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main_menu()
