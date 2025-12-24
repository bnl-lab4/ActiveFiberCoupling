"""
Continuously print the sensor reading until interrupted.
"""

import logging
from typing import Union
from MovementClasses import StageDevices

# unique logger name for this module
log = logging.getLogger(__name__)


def run(stage: StageDevices, exposureTime: Union[int, float], avg: bool = True):
    """
    Continuously print the sensor output with the supplied exposure time.

    Parameters
    ----------
    stage : `MovementClasses.StageDevices`
        The stage whose sensor will be read.
    exposureTime : int, float
        Exposure time for which to integrate the sensor.
    avg : bool, default=True
        Whether to average the sensor readout over the integration time.
    """
    log.info(
        f"Initiating continuous readout of {stage.name} sensor with"
        + f" exposure time {exposureTime} and avg = {avg}"
    )
    try:
        while True:
            intensity = stage.integrate(exposureTime, avg)
            print(intensity)
    except KeyboardInterrupt:
        return
