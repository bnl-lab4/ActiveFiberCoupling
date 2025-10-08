import time
import enum
import sigfig
import logging
import socket
import contextlib
import piplates.DAQC2plate as DAQ
from typing import Dict, Optional, Union

# unique logger name for this module
log = logging.getLogger(__name__)


class SensorType(enum.Enum):
    SIPM = "SiPM"
    PHOTODIODE = "photodiode"
    SOCKET = "socket"


class Sipm:
    def __init__(self, connection_dict: Dict[str, str]):
        self.addr = connection_dict['addr']
        self.channel = connection_dict['channel']
        DAQ.VerifyADDR(self.addr)
        DAQ.VerifyAINchannel(self.channel)
        log.info(f"Initialized to SiPM at addr {self.addr}, channel {self.channel}")

    def __enter__(self):
        pass    # not sure what to do here yet
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass    # not sure what to do here yet
        return False

    def read(self):
        # getADC has several waits in it, we could prbably slim it down
        power = DAQ.getADC(self.addr, self.channel)

        return power

    def integrate(self, Texp: Union[int, float], avg: bool = False):
        # this would be much better if it integrated over time
        power = 0
        Texp = int(Texp)
        for i in range(Texp):
            power += self.read()
            time.sleep(1e-5)
        if avg:
            power /= Texp
        log.debug(f"SiPM addr {self.addr} channel {self.channel} : " +
                f"integrated power {sigfig.round(power, 6)}{' averaged' if avg else ''} over {Texp} iterations")
        return power


class Socket:
    def __init__(self, connection_dict: Dict[str, str]):
        self.host = connection_dict['host']
        self.port = connection_dict['port']
        self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.connection.settimeout(20)
            self.connection.connect((self.host, self.port))
            log.info(f"Connected to socket host {self.host} at port {self.port}")
        except TimeoutError:
            log.warn(f"Timed out attempting to connect to host {self.host} at port {self.port}")
        self.exit_stack = contextlib.ExitStack()

    def __enter__(self):
        self.exit_stack.enter_context(self.connection)
        log.debug(f"Entered context of socket host {self.host} at port {self.port}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.exit_stack.close()
        log.info(f"Socket connection to host {self.host} at port {self.port} closed")
        log.debug(f"Exited context of socket host {self.host} at port {self.port}")

    def read(self):
        # getting a single reading does not make sense with the qutag
        # defaults to a 100ms integration time
        self.connection.sendall(str(100).encode('utf-8'))
        power = int(self.connection.recv(1024).decode('utf-8'))
        return power

    def integrate(self, Texp: Union[int, float], avg: bool = True):
        self.connection.sendall(str(Texp).encode('utf-8'))
        power = float(self.connection.recv(1024).decode('utf-8'))
        if not avg:
            power *= Texp
        power = int(power)
        log.debug(f"Socket at host {self.host} port {self.port} returned : " +
                f"{sigfig.round(power, 6)} {'averaged ' if avg else ''}over {Texp}ms")
        return power


class Photodiode:
    def __init__(self, connection_dict: Dict[str, str]):
        self.addr = connection_dict['addr']
        self.channel = connection_dict['channel']
        DAQ.VerifyADDR(self.addr)
        DAQ.VerifyAINchannel(self.channel)
        log.info(f"Initialized to photodiode at addr {self.addr}, channel {self.channel}")

    def __enter__(self):
        pass # no context management needed
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass    # no context management needed
        return False

    def read(self):
        # getADC has several waits in it, we could prbably slim it down
        power = DAQ.getADC(self.addr, self.channel)
        return power    #-power       # NOTE THE MINUS SIGN

    def integrate(self, Texp: Union[int, float], avg: bool = True):
        power = 0
        Texp = int(Texp)
        for i in range(Texp):
            power += self.read()
        if avg:
            power /= Texp
        log.debug(f"Photodiode addr {self.addr} channel {self.channel} : " +
                f"integrated power {sigfig.rounf(power, 6)}{' averaged' if avg else ''} over {Texp} interations")
        return power


class Sensor:

    def __init__(self, connection_dict: Dict[str, str],
                    sensorType: Optional[enum.Enum] = SensorType.PHOTODIODE):
        self._exit_stack = contextlib.ExitStack()
        self.connection_dict = connection_dict
        self.sensorType = sensorType
        if sensorType == SensorType.SIPM:
            self.sensor = Sipm(connection_dict)
        elif sensorType == SensorType.PHOTODIODE:
            self.sensor = Photodiode(connection_dict)
        elif sensorType == SensorType.SOCKET:
            self.sensor = Socket(connection_dict)
        else:
            raise ValueError("sensorType must be a SensorType enum.")

    def __enter__(self):
        self._exit_stack.enter_context(self.sensor)
        log.debug("Entered sensor context")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._exit_stack.close()
        log.debug("Exited context stack gracefully")
        return False

    def read(self):
        return self.sensor.read()

    def integrate(self, Texp: Union[int, float], avg: Optional[bool] = True):
        return self.sensor.integrate(Texp, avg)
