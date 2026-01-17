# ========================================================================
# TODO:
# Change Sipm and Photodiode integrate methods to count time, not iterations.
# Implement average in Socket.
# ========================================================================

"""
Defines the `Sensor` class and sensor-type subclasses.
"""

import contextlib
import enum
import socket
from typing import Dict, Optional, Union

import sigfig

from hardware_interfaces import DAQ
from logging_utils import get_logger

# unique logger name for this module
logger = get_logger(__name__)


class SensorType(enum.Enum):
    """
    An enumeration of the possible sensor types.

    See `Piplate` and `Socket` for info on the specific sensor types.

    Members
    -------
    PIPLATE : str
        Represents a sensor that is read via a Pi-Plate.
    SOCKET : str
        Represents a socket connection to another computer that acts as
        a sensor.
    """

    PIPLATE = "piplate"
    """Represents reading a Pi-Plate input for sensor data."""
    SOCKET = "socket"
    """Represents querying a socket server for sensor data."""


class Piplate:
    """
    Subclass of `Sensor` for reading from a Pi-plate.

    Reads in a single analog input channel on a DAQC2plate from Pi-Plates
    (<https://pi-plates.com/daqc2r1/>). This input can be integrated by
    calling the Pi-Plate read function many times.

    Parameters
    ----------
    connection_dict : Dict[str, str]
        A dictionary of the necessary info for reading out the voltage
        from the Pi-Plate. This must include the Pi-Plate address and
        channel (and the `SensorType` `PIPLATE` enum for `Sensor`).

    Methods
    -------
    read()
        Read the voltage of the analog input (once) .
    integrate(exposure_time, avg=True)
        Read the voltage of the Pi-Plate analog input exposure_time times and
        return the sum (default) or average value.

    Notes
    -----
    Inherits no-op ``__enter__`` and ``__exit__`` dunders from
    `Sensor` (parent).
    """

    def __init__(self, connection_dict: Dict[str, str]):
        self.addr = connection_dict["addr"]
        self.channel = connection_dict["channel"]
        DAQ.VerifyADDR(self.addr)
        DAQ.VerifyAINchannel(self.channel)
        logger.info(
            f"Initialized to Pi-Plate at addr {self.addr}, channel {self.channel}"
        )

    def read(self):
        """
        Read the voltage from the Pi-Plate input (once) .

        Returns
        -------
        float
            The value of the Pi-Plate input.

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
        # logger.trace(f"Pi-Plate addr {self.addr} channel {self.channel} : " +
        #             f"read power {sigfig.round(power, 6, warn=False)}")
        return power

    def integrate(self, exposure_time: Union[int, float], avg: bool = True):
        """
        Integrate the voltage of the Pi-plate  over many readings.

        Calls `read` `exposure_time` times and returns either the sum
        or the average (default) of the readings.

        Parameters
        ----------
        exposure_time : int, float
            Number of times to call read the voltage input.
        avg : bool, default=False
            Whether to average the total integrated value over the number
            of readings.

        Returns
        -------
        float
            The integrated value from the Pi-Plate.

        Notes
        -----
        Despite the parameter name `exposure_time`, the voltage will be read an
        integer number of times. However, each function call of
        `piplates.DAQC2plate.getADC` typically takes about 1 millisecond.
        """
        # this would be much better if it integrated over time
        power = 0
        exposure_time = int(exposure_time)
        for _ in range(exposure_time):
            power += self.read()
        if avg:
            power /= exposure_time
        logger.trace(
            f"Piplate addr {self.addr} channel {self.channel} : "
            + f"integrated power {sigfig.round(power, 6, warn=False)}{' averaged' if avg else ''} over {exposure_time} iterations"
        )
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
        from the Pi-Plate. This must include the host IP address and port
        (and the `SensorType` `SOCKET` enum for `Sensor`).

    Methods
    -------
    read()
        Get a sensor reading over 1 ms from the server.
    integrate(exposure_time, avg=True)
        Get a sensor reading from the socket server over `exposure_time`.

    Notes
    -----
    Overrides the no-op ``__enter__`` and ``__exit__`` dunders from `Sensor`.

    The other computer must be running a socket server and ready to
    ready to receive instructions from the client (the Pi). So far, this
    has been used to read the output from a SiPM using a quTAG from
    qutools (<https://qutools.com/qutag/>). The files ``qutag.py`` and
    ``get_timestamps.py`` are available for doing this on a windows machine.
    """

    def __init__(self, connection_dict: Dict[str, str]):
        self.host = connection_dict["host"]
        self.port = connection_dict["port"]
        self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.connection.settimeout(20)
            self.connection.connect((self.host, self.port))
            logger.info(f"Connected to socket host {self.host} at port {self.port}")
        except TimeoutError:
            logger.warning(
                f"Timed out attempting to connect to host {self.host} at port {self.port}"
            )
        self.exit_stack = contextlib.ExitStack()

    def __enter__(self):
        """
        Enables context management of the socket connection to the server.

        Returns self for use in 'with... as...' statements.
        """
        self.exit_stack.enter_context(self.connection)
        logger.debug(f"Entered context of socket host {self.host} at port {self.port}")
        return self

    def __exit__(self, _, __, ___):
        """
        Closes the connection to the socket server.

        Returns ``False``, letting exceptions propogate.
        """
        self.exit_stack.close()
        logger.info(f"Socket connection to host {self.host} at port {self.port} closed")
        logger.debug(f"Exited context of socket host {self.host} at port {self.port}")

    def read(self):
        """
        Get a sensor reading over 1 ms from the server.

        Returns
        -------
        int
            Value returned from the socket server.
        """
        # getting a single reading does not make sense with the qutag
        # uses 1ms integration time to be similar to other sensors
        self.connection.sendall(str(1).encode("utf-8"))
        power = int(self.connection.recv(1024).decode("utf-8"))
        # logger.trace(f"Socket at host {self.host} port {self.port} : " +
        #         f"returned {sigfig.round(power, 6, warn=False)} averaged over 100ms")
        return power

    def integrate(self, exposure_time: Union[int, float], avg: bool = True):
        """
        Get a sensor reading over `exposure_time` ms from the server.

        Parameters
        ----------
        exposure_time : int, float
            Number of milliseconds the server will integrate for.
        avg : bool, default=True
            Whether the server should return the sum (default) or average
            reading over `exposure_time`. Currently ignored and reserved for
            future use.

        Returns
        -------
        float
            The integrated value from the socket server.
        """
        self.connection.sendall(str(exposure_time).encode("utf-8"))
        power = float(self.connection.recv(1024).decode("utf-8"))
        # power = int(power)
        logger.trace(
            f"Socket at host {self.host} port {self.port} : "
            + f"returned {sigfig.round(power, 6, warn=False)} {'averaged ' if avg else ''}over {exposure_time}ms"
        )
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
    integrate(exposure_time, avg=True)
        Read the sensor over `exposure_time` and return the sum or average (default).

    Notes
    -----
    Provides no-op context management dunders ``__enter__`` and
    ``__exit__`` for subclass that do not need context management.
    """

    def __init__(
        self,
        connection_dict: Dict[str, str],
        sensor_type: Optional[enum.Enum] = SensorType.PIPLATE,
    ):
        self._exit_stack = contextlib.ExitStack()
        self.connection_dict = connection_dict
        self.sensor_type = sensor_type
        if sensor_type == SensorType.PIPLATE:
            self.sensor = Piplate(connection_dict)
        elif sensor_type == SensorType.SOCKET:
            self.sensor = Socket(connection_dict)
        else:
            raise ValueError("sensor_type must be a SensorType enum.")

    def __enter__(self):
        """
        No-op method for use with context management.

        Returns ``self`` for use in 'with... as...' statements.
        """
        pass
        logger.debug("Entered sensor context")
        return self

    def __exit__(self, _, __, ___):
        """
        No-op method for use with context management.

        Returns ``False``, letting exceptions propogate.
        """
        pass
        logger.debug("Exited context stack")
        return False

    def read(self):
        """
        Read the value of the sensor.

        See `Piplate` and `Socket` for more details.

        Returns
        -------
        float or int
            Value read from the sensor.
        """
        return self.sensor.read()

    def integrate(self, exposure_time: Union[int, float], avg: bool = True):
        """
        Returns the value of the sensor over `exposure_time`.

        See `Piplate` and `Socket` for more details.

        Parameters
        ----------
        Text : int, float
            How long to integrate for. Some sensors take this directly
            in milliseconds, while others iterate over this many sensor
            calls.
        avg : bool, default=True
            Whether to average the sensor output over the integration
            time. Ignored by some sensors.

        Returns
        -------
        float or int
            Integrated value from the sensor.
        """
        return self.sensor.integrate(exposure_time, avg)
