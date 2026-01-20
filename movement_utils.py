"""
Basic movement controls for steppers and piezos.
"""

from typing import Optional

from logging_utils import get_logger
from movement_classes import MovementType, StageDevices

# unique logger name for this module
logger = get_logger(__name__)


def energize(stage: StageDevices, axes: Optional[str] = None) -> None:
    """
    Energize one or more steppers. See movement_classes.StageAxis.energize.

    Parameters
    ----------
    stage : `movement_classes.StageDevices`
        The stage for which the steppers will be energized.
    axes : str, optional
        Which axes' steppers to energize, all listed in a single
        string (e.g. 'xz'), or 'all' for all axes. ``None``
        defaults to 'all'.
    """
    stage.energize(axes)


def home(stage: StageDevices, axes: Optional[str] = None) -> None:
    """
    Home one or more steppers. See movement_classes.StageAxis.home.

    Parameters
    ----------
    stage : `movement_classes.StageDevices`
        The stage for which the steppers will be homed.
    axes : str, optional
        Which axes' steppers to home, all listed in a single
        string (e.g. 'xz'), or 'all' for all axes. ``None``
        defaults to 'all'.
    """
    stage.home(axes)


def deenergize(stage: StageDevices, axes: Optional[str] = None) -> None:
    """
    Deenergize one or more stepper. See movement_classes.StageAxis.deenergize.

    Parameters
    ----------
    stage : `movement_classes.StageDevices`
        The stage for which the steppers will be deenergized.
    axes : str, optional
        Which axes' steppers to deenergize, all listed in a single
        string (e.g. 'xz'), or 'all' for all axes. ``None``
        defaults to 'all'.
    """
    stage.deenergize(axes)


def center(stage: StageDevices, which: MovementType) -> None:
    """
    Center all piezos' and/or steppers' positions.

    Move the piezos and/or steppers of the stage to the center of their respective travel ranges.

    Parameters
    ----------
    stage : `movement_classes.StageDevices`
        The stage for which the axes will be centered.
    which : `movement_classes.MovementType`
        Whether to move the piezos, the steppers, or both (``GENERAL``).
    """

    if which == MovementType.PIEZO or which == MovementType.GENERAL:
        logger.info(f"Centering {stage.name} piezos")
        stage.goto("x", stage.axes["x"].piezo_center, MovementType.PIEZO)
        stage.goto("y", stage.axes["y"].piezo_center, MovementType.PIEZO)
        stage.goto("z", stage.axes["z"].piezo_center, MovementType.PIEZO)
    if which == MovementType.STEPPER or which == MovementType.GENERAL:
        logger.info(f"Centering {stage.name} steppers")
        # empty assignments makes them finish before moving on
        _ = stage.goto("x", stage.axes["x"].stepper_center, MovementType.STEPPER)
        _ = stage.goto("y", stage.axes["y"].stepper_center, MovementType.STEPPER)
        _ = stage.goto("z", stage.axes["z"].stepper_center, MovementType.STEPPER)


def zero(stage: StageDevices, which: MovementType) -> None:
    """
    Zero all piezos' and/or steppers' positions.

    Move the piezos and/or steppers of the stage to the minimum of their respective travel ranges.

    Parameters
    ----------
    stage : `movement_classes.StageDevices`
        The stage for which the axes will be zeroed.
    which : `movement_classes.MovementType`
        Whether to move the piezos, the steppers, or both (``GENERAL``).
    """
    if which == MovementType.PIEZO or which == MovementType.GENERAL:
        logger.info(f"Zeroing {stage.name} piezos")
        stage.goto("x", stage.axes["x"].piezo_limits[0], MovementType.PIEZO)
        stage.goto("y", stage.axes["y"].piezo_limits[0], MovementType.PIEZO)
        stage.goto("z", stage.axes["z"].piezo_limits[0], MovementType.PIEZO)
    if which == MovementType.STEPPER or which == MovementType.GENERAL:
        logger.info(f"Zeroing {stage.name} steppers")
        # empty assignments makes them finish before moving on
        _ = stage.goto("x", stage.axes["x"].stepper_limits[0], MovementType.STEPPER)
        _ = stage.goto("y", stage.axes["y"].stepper_limits[0], MovementType.STEPPER)
        _ = stage.goto("z", stage.axes["z"].stepper_limits[0], MovementType.STEPPER)


def max(stage: StageDevices, which: MovementType) -> None:
    """
    Maximize all piezos' and/or steppers' positions.

    Move the piezos and/or steppers of the stage to the maximum of their respective travel ranges.

    Parameters
    ----------
    stage : `movement_classes.StageDevices`
        The stage for which the axes will be maximized.
    which : `movement_classes.MovementType`
        Whether to move the piezos, the steppers, or both (``GENERAL``).
    """

    if which == MovementType.PIEZO or which == MovementType.GENERAL:
        logger.info(f"Maximizing {stage.name} piezos")
        stage.goto("x", stage.axes["x"].piezo_limits[1], MovementType.PIEZO)
        stage.goto("y", stage.axes["y"].piezo_limits[1], MovementType.PIEZO)
        stage.goto("z", stage.axes["z"].piezo_limits[1], MovementType.PIEZO)
    if which == MovementType.STEPPER or which == MovementType.GENERAL:
        logger.info(f"Maximizing {stage.name} steppers")
        # empty assignments makes them finish before moving on
        _ = stage.goto("x", stage.axes["x"].stepper_limits[1], MovementType.STEPPER)
        _ = stage.goto("y", stage.axes["y"].stepper_limits[1], MovementType.STEPPER)
        _ = stage.goto("z", stage.axes["z"].stepper_limits[1], MovementType.STEPPER)
