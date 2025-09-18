import time
from ticlib import TicUSB

stepper = TicUSB(product = 0x00b5, serial_number = '00485159')

stepper.halt_and_set_position(0)
stepper.energize()
stepper.exit_safe_start()

stepper.set_target_position(100)
while stepper.get_target_position() != stepper.get_current_position():
    time.sleep(0.1)

stepper.deenergize()
stepper.enter_safe_start()
