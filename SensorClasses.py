# ========================================================================
# TODO:
# Move Sipm and Photodiode context dunders to Sensor.
# Combine Sipm and Photodiode into a single PiPlate class.
# Change Sipm and Photodiode integrate methods to count time, not iterations.
# Implement average in Socket.
# ========================================================================

"""
Defines the `Sensor` class and sensor-type subclasses.
"""
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
    """
    An enumeration of the possible sensor types.

    See `Sipm`, `Photodiode` and `Socket` for info on the specific
    sensor types.

    Members
    -------
    SIPM : str
        Represents a SiPM plugged into a Pi-Plate.
    PHOTODIODE : str
        Represents a photodiode plugged into a Pi-Plate.
    SOCKET : str
        Represents a socket connection to another computer that acts as
        a sensor.
    """

    SIPM = "SiPM"
    """Represents a SiPM sensor."""
    PHOTODIODE = "photodiode"
    """Represents a photodiode sensor."""
    SOCKET = "socket"
    """Represenst a socket connection querried as a sensor."""


class Sipm:
    """
    Subclass of `Sensor` for a SiPM device.

    The output of a SiPM sensor device is connected to an analog input
    on a DAQC2plate from Pi-Plates (<https://pi-plates.com/daqc2r1/>).
    This input can be read once or integrated over (currently integrate via
    many sequential reads).

    Parameters
    ----------
    connection_dict : Dict[str, str]
        A dictionary of the necessary info for reading out the voltage
        from the Pi-Plate. This will include the Pi-Plate address and channel.

    Methods
    -------
    read()
        Read the voltage (once) of the analog input.
    integrate(Texp, avg=False)
        Read the voltage of the Pi-Plate analog input Texp times and
        return the sum (default) or average value.
    """
    def __init__(self, connection_dict: Dict[str, str]):
        self.addr = connection_dict['addr']
        self.channel = connection_dict['channel']
        DAQ.VerifyADDR(self.addr)
        DAQ.VerifyAINchannel(self.channel)
        log.info(f"Initialized to SiPM at addr {self.addr}, channel {self.channel}")

    def __enter__(self):
        """
        No-op method to conform to context manager protocol.
        
        Returns self for use in 'with... as...' statements.
        """
        pass # no context management needed
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        No-op method to conform to context manager protocol.
        
        Returns ``False``, letting exceptions propogate.
        """
        pass # no context management needed
        return False

    def read(self):
        """
        Read the voltage (once) of the SiPM from the Pi-Plate input.

        Notes
        -----
        It is currently unclear whether the value returned is volts or
        some other quantity. Additionally, it is not clear whether there
        is just an unknown zero point and/or a non-linear relationship
        between the input voltage and the returned ADC value, though the
        relationship seems to be monotonic.
        """
        # getADC has several waits in it, we could prbably slim it down
        power = DAQ.getADC(self.addr, self.channel)
        # log.trace(f"SiPM addr {self.addr} channel {self.channel} : " +
        #             f"read power {sigfig.round(power, 6, warn=False)}")
        return power

    def integrate(self, Texp: Union[int, float], avg: bool = False):
        """
        Integrate the voltage of the SiPM over many readings.

        Calls `read` `Texp` times and returns either the sum (default)
        or the average of the readings.

        Parameters
        ----------
        Texp : int, float
            Number of times to call read the voltage input.
        avg : bool, default=False
            Whether to average the total integrated value over the number
            of readings.

        Notes
        -----
        Despite the parameter name `Texp`, the voltage will be read an
        integer number of times. However, each function call of
        `piplates.DAQC2plate.getADC` typically takes about 1 millisecond.
        """
        # this would be much better if it integrated over time
        power = 0
        Texp = int(Texp)
        for i in range(Texp):
            power += self.read()
        if avg:
            power /= Texp
        log.trace(f"SiPM addr {self.addr} channel {self.channel} : " +
                f"integrated power {sigfig.round(power, 6, warn=False)}{' averaged' if avg else ''} over {Texp} iterations")
        return power


class Socket:
    """
    Subclass of `Sensor` for a socket connection querried as a sensor.

    Instead of connecting a sensor directly to the Raspberry Pi, the
    sensor can be connected to another computer that then sends its
    readings to the Pi through a socket connection. The sensor's computer
    hosts a socket server which the Pi establishes a connection to.

    Parameters
    ----------
    connection_dict : Dict[str, str]
        A dictionary of the necessary info for reading out the voltage
        from the Pi-Plate. This will include the host IP address and port.

    Methods
    -------
    read()
        Get a sensor reading over 1 ms from the server.
    integrate(Texp, avg=True)
        Get a sensor reading from the socket server over `Texp`.

    Notes
    -----
    The other computer must be running a socket server and ready to
    ready to receive instructions from the client (the Pi). So far, this
    has been used to read the output from a SiPM using a quTAG from
    qutools (<https://qutools.com/qutag/>). The files ``qutag.py`` and
    ``get_timestamps.py`` are available for doing this on a windows machine.
    """

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
        """
        Enables context management of the socket connection to the server.

        Returns self for use in 'with... as...' statements.
        """
        self.exit_stack.enter_context(self.connection)
        log.debug(f"Entered context of socket host {self.host} at port {self.port}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Closes the connection to the socket server.

        Returns ``False``, letting exceptions propogate.
        """
        self.exit_stack.close()
        log.info(f"Socket connection to host {self.host} at port {self.port} closed")
        log.debug(f"Exited context of socket host {self.host} at port {self.port}")

    def read(self):
        """
        Get a sensor reading over 1 ms from the server.
        """
        # getting a single reading does not make sense with the qutag
        # uses 1ms integration time to be similar to other sensors
        self.connection.sendall(str(1).encode('utf-8'))
        power = int(self.connection.recv(1024).decode('utf-8'))
        # log.trace(f"Socket at host {self.host} port {self.port} : " +
        #         f"returned {sigfig.round(power, 6, warn=False)} averaged over 100ms")
        return power

    def integrate(self, Texp: Union[int, float], avg: bool = True):
        """
        Get a sensor reading over `Texp` ms from the server.

        Parameters
        ----------
        Texp : int, float
            Number of milliseconds the server will integrate for.
        avg : bool, default=True
            Whether the server should return the sum (default) or average
            reading over `Texp`. Currently ignored and reserved for
            future use.
        """
        self.connection.sendall(str(Texp).encode('utf-8'))
        power = float(self.connection.recv(1024).decode('utf-8'))
        power = int(power)
        log.trace(f"Socket at host {self.host} port {self.port} : " +
                f"returned {sigfig.round(power, 6, warn=False)} {'averaged ' if avg else ''}over {Texp}ms")
        return power


class Photodiode:
    """
    Subclass of `Sensor` for a photodiode, possibly amplified.

    The output of a photodiode or amplifier is connected to an analog input
    on a DAQC2plate from Pi-Plates (<https://pi-plates.com/daqc2r1/>).
    This input can be read once or integrated over (currently integrates via
    many sequential reads).

    Parameters
    ----------
    connection_dict : Dict[str, str]
        A dictionary of the necessary info for reading out the voltage
        from the Pi-Plate. Must include the Pi-Plate address and channel.

    Methods
    -------
    read()
        Read the voltage (once) of the analog input.
    integrate(Texp, avg=False)
        Read the voltage of the Pi-Plate analog input Texp times and
        return the sum (default) or average value.
    """
    def __init__(self, connection_dict: Dict[str, str]):
        self.addr = connection_dict['addr']
        self.channel = connection_dict['channel']
        DAQ.VerifyADDR(self.addr)
        DAQ.VerifyAINchannel(self.channel)
        log.info(f"Initialized to photodiode at addr {self.addr}, channel {self.channel}")

    def __enter__(self):
        """
        No-op method to conform to context manager protocol.
        
        Returns self for use in 'with... as...' statements.
        """
        pass # no context management needed
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        No-op method to conform to context manager protocol.
        
        Returns ``False``, letting exceptions propogate.
        """
        pass    # no context management needed
        return False

    def read(self):
        """
        Read the voltage (once) of the photodiode from the Pi-Plate input.

        Notes
        -----
        It is currently unclear whether the value returned is volts or
        some other quantity. Additionally, it is not clear whether there
        is just an unknown zero point and/or a non-linear relationship
        between the input voltage and the returned ADC value, though the
        relationship seems to be monotonic.
        """
        # getADC has several waits in it, we could prbably slim it down
        power = DAQ.getADC(self.addr, self.channel)
        # log.trace(f"Photodiode addr {self.addr} channel {self.channel} : " +
        #             f"read power {sigfig.round(power, 6, warn=False)}")
        return power

    def integrate(self, Texp: Union[int, float], avg: bool = False):
        """
        Integrate the voltage of the photodiode over many readings.

        Calls `read` `Texp` times and returns either the sum (default)
        or the average of the readings.

        Parameters
        ----------
        Texp : int, float
            Number of times to call read the voltage input.
        avg : bool, default=False
            Whether to average the total integrated value over the number
            of readings.

        Notes
        -----
        Despite the parameter name `Texp`, the voltage will be read an
        integer number of times. However, each function call of
        ``piplates.DAQC2plate.getADC`` typically takes about 1 millisecond.
        """
        power = 0
        Texp = int(Texp)
        for i in range(Texp):
            power += self.read()
        if avg:
            power /= Texp
        log.trace(f"Photodiode addr {self.addr} channel {self.channel} : " +
                f"integrated power {sigfig.rounf(power, 6)}{' averaged' if avg else ''} over {Texp} interations")
        return power


class Sensor:
    """
    Parent class for sensors.

    Parameters
    ----------
    connection_dict : Dict[str, str]
        A dictionary of the necessary info for reading out the sensor.
        Must contain the sensor type as a `SensorType` enumeration.

    Methods
    -------
    read()
        Read the sensor once.
    integrate(Texp, avg=True)
        Read the sensor over `Texp` and return the sum or average (default).
    """

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
        """
        Enter the context of the sensor.

        Returns self for use in 'with... as...' statements.
        """
        self._exit_stack.enter_context(self.sensor)
        log.debug("Entered sensor context")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the context of the sensor.

        Returns ``False``, letting exceptions propogate.
        """
        self._exit_stack.close()
        log.debug("Exited context stack")
        return False

    def read(self):
        """
        Read the value of the sensor.

        Refer to the sensor subclasses for more details.
        """
        return self.sensor.read()

    def integrate(self, Texp: Union[int, float], avg: Optional[bool] = True):
        """
        Returns the value of the sensor over `Texp`.

        Refer to the sensor subclasses for more details.

        Parameters
        ----------
        Text : int, float
            How long to integrate for. Some sensors take this directly
            in milliseconds, while others iterate over this many sensor
            calls.
        avg : bool, default=True
            Whether to average the sensor output over the integration
            time. Ignored by some sensors.
        """
        return self.sensor.integrate(Texp, avg)
