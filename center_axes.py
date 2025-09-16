from MovementClasses import StageDevices, Position, MovementType

PIEZO_CENTER = Position(37.5, 'volts')
STEPPER_CENTER = Position(200.0, 'steps')

def run(stage: StageDevices, which: MovementType):
    if which == MovementType.PIEZO or which == MovementType.GENERAL:
        stage.goto('x', PIEZO_CENTER, MovementType.PIEZO)
        stage.goto('y', PIEZO_CENTER, MovementType.PIEZO)
        stage.goto('z', PIEZO_CENTER, MovementType.PIEZO)
    if which == MovementType.STEPPER or which == MovementType.GENERAL:
        stage.goto('x', STEPPER_CENTER, MovementType.STEPPER)
        stage.goto('y', STEPPER_CENTER, MovementType.STEPPER)
        stage.goto('z', STEPPER_CENTER, MovementType.STEPPER)

