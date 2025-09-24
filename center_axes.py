from MovementClasses import StageDevices, MovementType
import logging

# unique logger name for this module
log = logging.getLogger(__name__)


def run(stage: StageDevices, which: MovementType):
    if which == MovementType.PIEZO or which == MovementType.GENERAL:
        log.info(f"Centering {stage.name} piezos")
        stage.goto('x', stage.axes['x'].PIEZO_CENTER, MovementType.PIEZO)
        stage.goto('y', stage.axes['y'].PIEZO_CENTER, MovementType.PIEZO)
        stage.goto('z', stage.axes['z'].PIEZO_CENTER, MovementType.PIEZO)
    if which == MovementType.STEPPER or which == MovementType.GENERAL:
        log.info(f"Centering {stage.name} steppers")
        # empty assignments makes them finish before moving on
        _ = stage.goto('x', stage.axes['x'].STEPPER_CENTER, MovementType.STEPPER)
        _ = stage.goto('y', stage.axes['y'].STEPPER_CENTER, MovementType.STEPPER)
        _ = stage.goto('z', stage.axes['z'].STEPPER_CENTER, MovementType.STEPPER)
