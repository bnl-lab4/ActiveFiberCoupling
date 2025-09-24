from MovementClasses import StageDevices, MovementType
import logging

# unique logger name for this module
log = logging.getLogger(__name__)


def run(stage: StageDevices, which: MovementType):
    if which == MovementType.PIEZO or which == MovementType.GENERAL:
        log.info(f"Zeroing {stage.name} piezos")
        stage.goto('x', stage.axes['x'].PIEZO_LIMITS[0], MovementType.PIEZO)
        stage.goto('y', stage.axes['y'].PIEZO_LIMITS[0], MovementType.PIEZO)
        stage.goto('z', stage.axes['z'].PIEZO_LIMITS[0], MovementType.PIEZO)
    if which == MovementType.STEPPER or which == MovementType.GENERAL:
        log.info(f"Zeroing {stage.name} steppers")
        # empty assignments makes them finish before moving on
        _ = stage.goto('x', stage.axes['x'].STEPPER_LIMITS[0], MovementType.STEPPER)
        _ = stage.goto('y', stage.axes['y'].STEPPER_LIMITS[0], MovementType.STEPPER)
        _ = stage.goto('z', stage.axes['z'].STEPPER_LIMITS[0], MovementType.STEPPER)
