from MovementClasses import StageDevices, StageAxis, Distance, MovementType

def run(stage: StageDevices, which: MovementType):
    if which == MovementType.PIEZO or which == MovementType.GENERAL:
        stage.goto('x', stage.axes['x'].PIEZO_CENTER, MovementType.PIEZO)
        stage.goto('y', stage.axes['y'].PIEZO_CENTER, MovementType.PIEZO)
        stage.goto('z', stage.axes['z'].PIEZO_CENTER, MovementType.PIEZO)
    if which == MovementType.STEPPER or which == MovementType.GENERAL:
        _ = stage.goto('x', stage.axes['x'].STEPPER_CENTER, MovementType.STEPPER)
        _ = stage.goto('y', stage.axes['y'].STEPPER_CENTER, MovementType.STEPPER)
        _ = stage.goto('z', stage.axes['z'].STEPPER_CENTER, MovementType.STEPPER)

