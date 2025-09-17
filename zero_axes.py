from MovementClasses import StageDevices, StageAxis, Distance, MovementType

PIEZO_CENTER = StageAxis.PIEZO_CENTER
STEPPER_CENTER = StageAxis.STEPPER_CENTER

def run(stage: StageDevices, which: MovementType):
    if which == MovementType.PIEZO or which == MovementType.GENERAL:
        stage.goto('x', PIEZO_ZERO, MovementType.PIEZO)
        stage.goto('y', PIEZO_ZERO, MovementType.PIEZO)
        stage.goto('z', PIEZO_ZERO, MovementType.PIEZO)
    if which == MovementType.STEPPER or which == MovementType.GENERAL:
        stage.goto('x', STEPPER_ZERO, MovementType.STEPPER)
        stage.goto('y', STEPPER_ZERO, MovementType.STEPPER)
        stage.goto('z', STEPPER_ZERO, MovementType.STEPPER)
