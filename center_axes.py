from MovementClasses import Position, MovementType

PIEZO_CENTER = Position(37.5, 'volts')
STEPPER_CENTER = Position(200.0, 'steps')

def run(stage, which):
    if which == 'piezos':
        stage.goto('x', PIEZO_CENTER, MovementType.PIEZO)
        stage.goto('y', PIEZO_CENTER, MovementType.PIEZO)
        stage.goto('z', PIEZO_CENTER, MovementType.PIEZO)
    if which == 'steppers':
        stage.goto('x', STEPPER_CENTER, MovementType.STEPPER)
        stage.goto('y', STEPPER_CENTER, MovementType.STEPPER)
        stage.goto('z', STEPPER_CENTER, MovementType.STEPPER)

