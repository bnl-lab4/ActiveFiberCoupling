from MovementClasses import Position, MovementType

PIEZO_ZERO = Position(0.0, 'volts')
STEPPER_ZERO = Position(0.0, 'steps')

def run(stage, which):
    if which == 'piezos':
        stage.goto('x', PIEZO_ZERO, MovementType.PIEZO)
        stage.goto('y', PIEZO_ZERO, MovementType.PIEZO)
        stage.goto('z', PIEZO_ZERO, MovementType.PIEZO)
    if which == 'steppers':
        stage.goto('x', STEPPER_ZERO, MovementType.STEPPER)
        stage.goto('y', STEPPER_ZERO, MovementType.STEPPER)
        stage.goto('z', STEPPER_ZERO, MovementType.STEPPER)
