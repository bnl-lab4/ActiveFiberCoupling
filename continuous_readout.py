"""
Continuously print the sensor reading until interrupted.
"""

from typing import Union

from logging_utils import get_logger
from movement_classes import StageDevices

# unique logger name for this module
logger = get_logger(__name__)


def run(stage: StageDevices, exposure_time: Union[int, float], avg: bool = True):
    """
    Continuously print the sensor output with the supplied exposure time.

    Parameters
    ----------
    stage : `movement_classes.StageDevices`
        The stage whose sensor will be read.
    exposure_time : int, float
        Exposure time for which to integrate the sensor.
    avg : bool, default=True
        Whether to average the sensor readout over the integration time.
    """
    logger.info(
        f"Initiating continuous readout of {stage.name} sensor with"
        + f" exposure time {exposure_time} and avg = {avg}"
    )
    try:
        while True:
            intensity = stage.integrate(exposure_time, avg)
            print(intensity)
    except KeyboardInterrupt:
        return
