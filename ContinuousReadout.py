import logging
from typing import Union
from MovementClasses import StageDevices

# unique logger name for this module
log = logging.getLogger(__name__)


def run(stage: StageDevices, exposureTime: Union[int, float], avg: bool = True):
    log.info(f"Initiating continuous readout of {stage.name} sensor with" +
             f" exposure time {exposureTime} and avg = {avg}")
    try:
        while True:
            intensity = stage.integrate(exposureTime, avg)
            print(intensity)
    except KeyboardInterrupt:
        return
