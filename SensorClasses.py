#### it would be nice to also have a socket class, but that would require a
####    fundamentally different structure, effecting changes all the up to main.py
import enum
import time
import logging
import warnings
import socket
import piplates.DAQC2plate as DAQ
from typing import Dict, Optional

# unique logger name for this module
log = logging.getLogger(__name__)

class SensorType(enum.Enum):
    SIPM = "SiPM"
    PHOTODIODE = "photodiode"
    # SOCKET = "socket"


class Sipm:
    def __init__(self, connection_dict: Dict[str, str]):
        self.addr = connection_dict['addr']
        self.channel = connection_dict['channel']
        DAQ.VerifyADDR(self.addr)
        DAQ.VerifyAINchannel(self.channel)
        log.info(f"Initialized to SiPM at addr {self.addr}, channel {self.channel}")
        return

    def read(self):
        # getADC has several waits in it, we could prbably slim it down
        power = DAQ.getADC(self.addr, self.channel)
        log.debug(f"SiPM addr:{self.addr} channel:{self.channel} read power {power:.6f}")
        return power

    def integrate(self, Texp: int, avg: bool = True):
        # this would be much better if it integrated over time
        power = 0
        for i in range(Texp):
            power += self.read()
        if avg:
            power /= Texp
        log.info(f"SiPM addr:{self.addr} channel:{self.channel}  " +\
                f"integrated power {power:.6f}{' averaged' if avg else ''} over {Texp} interations")
        return power


class Photodiode:
    def __init__(self, connection_dict: Dict[str, str]):
        self.addr = connection_dict['addr']
        self.channel = connection_dict['channel']
        DAQ.VerifyADDR(self.addr)
        DAQ.VerifyAINchannel(self.channel)
        log.info(f"Initialized to photodiode at addr {self.addr}, channel {self.channel}")
        return

    def read(self):
        # getADC has several waits in it, we could prbably slim it down
        power = DAQ.getADC(self.addr, self.channel)
#       log.debug(f"Photodiode addr:{self.addr} channel:{self.channel} read power {power:.6f}")
        return -power       # NOTE THE MINUS SIGN

    def integrate(self, Texp: int, avg: bool = True):
        power = 0
        for i in range(Texp):
            power += self.read()
        if avg:
            power /= Texp
        log.debug(f"Photodiode addr:{self.addr} channel {self.channel}  " +\
                f"integrated power {power:.6f}{' averaged' if avg else ''} over {Texp} interations")
        return power



class Sensor:

    def __init__(self, connection_dict: Dict[str, str],
                    sensorType: Optional[enum.Enum] = SensorType.PHOTODIODE):
        self.connection_dict = connection_dict
        self.sensorType = sensorType
        if sensorType == SensorType.SIPM:
            self.sensor = Sipm(connection_dict)
        elif sensorType == SensorType.PHOTODIODE:
            self.sensor = Photodiode(connection_dict)
#       if sensorType == SensorType.SOCKET:
#           self.sensor = Socket(connection_dict)
        else:
            raise ValueError("sensorType must be a SensorType enum.")
        return

    def read(self):
        return self.sensor.read()

    def integrate(self, Texp: int, avg: Optional[bool] = True):
        return self.sensor.integrate(Texp, avg)
