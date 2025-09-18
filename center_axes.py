from MovementClasses import StageDevices, StageAxis, Distance, MovementType

PIEZO_CENTER = StageAxis.PIEZO_CENTER

def run(stage: StageDevices, which: MovementType):
    if which == MovementType.PIEZO or which == MovementType.GENERAL:
        stage.goto('x', PIEZO_CENTER, MovementType.PIEZO)
        stage.goto('y', PIEZO_CENTER, MovementType.PIEZO)
        stage.goto('z', PIEZO_CENTER, MovementType.PIEZO)
    if which == MovementType.STEPPER or which == MovementType.GENERAL:
        stage.goto('x', stage.axes['x'].STEPPER_CENTER, MovementType.STEPPER)
        stage.goto('y', stage.axes['y'].STEPPER_CENTER, MovementType.STEPPER)
        stage.goto('z', stage.axes['z'].STEPPER_CENTER, MovementType.STEPPER)

