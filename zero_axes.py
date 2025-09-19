from MovementClasses import StageDevices, StageAxis, Distance, MovementType

def run(stage: StageDevices, which: MovementType):
    if which == MovementType.PIEZO or which == MovementType.GENERAL:
        stage.goto('x', stage.axes['x'].PIEZO_LIMITS[0], MovementType.PIEZO)
        stage.goto('y', stage.axes['y'].PIEZO_LIMITS[0], MovementType.PIEZO)
        stage.goto('z', stage.axes['z'].PIEZO_LIMITS[0], MovementType.PIEZO)
    if which == MovementType.STEPPER or which == MovementType.GENERAL:
        _ = stage.goto('x', stage.axes['x'].STEPPER_LIMITS[0], MovementType.STEPPER)
        _ = stage.goto('y', stage.axes['y'].STEPPER_LIMITS[0], MovementType.STEPPER)
        _ = stage.goto('z', stage.axes['z'].STEPPER_LIMITS[0], MovementType.STEPPER)
