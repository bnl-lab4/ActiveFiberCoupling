from typing import Optional
from MovementClasses import StageDevices


def energize(stage: StageDevices, axes: Optional[str] = None):
    stage.energize(axes)


def home(stage: StageDevices, axes: Optional[str] = None):
    stage.home(axes)


def deenergize(stage: StageDevices, axes: Optional[str] = None):
    stage.deenergize(axes)
