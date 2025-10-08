import logging
from typing import Optional
from MovementClasses import StageDevices, MovementType

# unique logger name for this module
log = logging.getLogger(__name__)


def energize(stage: StageDevices, axes: Optional[str] = None):
    stage.energize(axes)


def home(stage: StageDevices, axes: Optional[str] = None):
    stage.home(axes)


def deenergize(stage: StageDevices, axes: Optional[str] = None):
    stage.deenergize(axes)


def center(stage: StageDevices, which: MovementType):
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


def zero(stage: StageDevices, which: MovementType):
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


def max(stage: StageDevices, which: MovementType):
    if which == MovementType.PIEZO or which == MovementType.GENERAL:
        log.info(f"Maximizing {stage.name} piezos")
        stage.goto('x', stage.axes['x'].PIEZO_LIMITS[1], MovementType.PIEZO)
        stage.goto('y', stage.axes['y'].PIEZO_LIMITS[1], MovementType.PIEZO)
        stage.goto('z', stage.axes['z'].PIEZO_LIMITS[1], MovementType.PIEZO)
    if which == MovementType.STEPPER or which == MovementType.GENERAL:
        log.info(f"Maximizing {stage.name} steppers")
        # empty assignments makes them finish before moving on
        _ = stage.goto('x', stage.axes['x'].STEPPER_LIMITS[1], MovementType.STEPPER)
        _ = stage.goto('y', stage.axes['y'].STEPPER_LIMITS[1], MovementType.STEPPER)
        _ = stage.goto('z', stage.axes['z'].STEPPER_LIMITS[1], MovementType.STEPPER)
