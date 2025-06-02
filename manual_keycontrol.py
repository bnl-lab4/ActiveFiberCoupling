import time
import curses
from motion import move

def run(ser):
    position = {'x': 0.0, 'y': 0.0, 'z': 0.0}

    try:
        increment = float(input("Enter movement increment (e.g., 0.05): "))
    except ValueError:
        print("Invalid increment.")
        return

    def control_loop(stdscr):
        stdscr.nodelay(True)
        stdscr.clear()
        stdscr.addstr("Control with keys:\n")
        stdscr.addstr("  X axis: a/d\n")
        stdscr.addstr("  Y axis: q/z\n")
        stdscr.addstr("  Z axis: w/x\n")
        stdscr.addstr("Press 'ESC' to exit.\n\n")
        stdscr.refresh()

        while True:
            key = stdscr.getch()
            moved = False

            if key == ord('a'):
                position['x'] -= increment
                move('x', position['x'], ser)
                moved = True
            elif key == ord('d'):
                position['x'] += increment
                move('x', position['x'], ser)
                moved = True
            elif key == ord('q'):
                position['y'] += increment
                move('y', position['y'], ser)
                moved = True
            elif key == ord('z'):
                position['y'] -= increment
                move('y', position['y'], ser)
                moved = True
            elif key == ord('w'):
                position['z'] += increment
                move('z', position['z'], ser)
                moved = True
            elif key == ord('x'):
                position['z'] -= increment
                move('z', position['z'], ser)
                moved = True
            elif key == 27:
                break

            if moved:
                stdscr.clear()
                stdscr.addstr("Control with keys:\n")
                stdscr.addstr("  X axis: a/d\n")
                stdscr.addstr("  Y axis: q/z\n")
                stdscr.addstr("  Z axis: w/x\n")
                stdscr.addstr("Press 'ESC' to exit.\n\n")
                stdscr.addstr(f"X: {position['x']:.3f}\n")
                stdscr.addstr(f"Y: {position['y']:.3f}\n")
                stdscr.addstr(f"Z: {position['z']:.3f}\n")
                stdscr.refresh()

            time.sleep(0.05)

    curses.wrapper(control_loop)
