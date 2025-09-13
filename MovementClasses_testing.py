import MovementClasses

SERIAL_PORT0 = '/dev/ttyACM0'

stepper_ports = dict(x = None, y = None, z = None)
stage = MovementClasses.StageDevices("stage_0", SERIAL_PORT0, stepper_ports)

# print(stage.axes)

result = stage.move('z', MovementClasses.Position(33.22, 'volts'), which=MovementClasses.MovementType.GENERAL)
print(result)
