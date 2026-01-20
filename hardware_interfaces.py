"""
Handles imports for hardware libraries.

If libraries are missing (e.g. on Windows/WSL2), it provides stubs
that raise errors only when accessed, allowing the file to be imported safely.
"""

from typing import Optional


class HardwareLibMissingStub:
    """
    Generic stub class that raises an error if you try to use it.

    Parameters
    ----------
    lib_name : str
        Name of library to provide a stub for.
    obj_name : str, optional
        Name of object in library to provide stub for.
    """

    def __init__(self, lib_name: str, obj_name: Optional[str] = None) -> None:
        self.lib_name = lib_name
        self.obj_name = obj_name or lib_name

    def __getattr__(self, name: str):
        """
        Raises a RuntimeError upon any attempt to access an attribute.

        Parameters
        ----------
        name : str
            Name of the attribute that was expected.
        """
        raise RuntimeError(
            f"Hardware Error: Cannot access '{self.obj_name}.{name}'. "
            f"The '{self.lib_name}' library is not installed. "
            "Ensure you are running on the Raspberry Pi or using Simulation classes."
        )

    def __call__(self, *_, **__):
        """Raises a RuntimeError upon any attempt to call stub."""
        raise RuntimeError(
            f"Hardware Error: Cannot initialize '{self.obj_name}'. "
            f"The '{self.lib_name}' library is not installed. "
            "Ensure you are running on the Raspberry Pi or using Simulation classes."
        )


# Import hardware libraries that may require stubs
try:
    import piplates.DAQC2plate as DAQ
except (ImportError, ModuleNotFoundError):
    DAQ = HardwareLibMissingStub("piplates", "DAQC2plate")

try:
    from ticlib import TicUSB
except (ImportError, ModuleNotFoundError):
    TicUSB = HardwareLibMissingStub("ticlib", "TicUSB")

try:
    import serial

    Serial = serial.Serial
    SerialException = serial.SerialException
except (ImportError, ModuleNotFoundError):
    Serial = HardwareLibMissingStub("pyserial", "serial.Serial")

    # Define stub SerialExcpetion, which is expected in some places
    class SerialException(Exception):
        pass

    raise SerialException


__all__ = ["DAQ", "TicUSB", "Serial", "SerialException"]
