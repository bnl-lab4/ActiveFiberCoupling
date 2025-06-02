# ActiveFiberCoupling
Automatic fiber alignment and coupling using the output of a photodiode and a 3-axis stage

Needed equipment:
    * Raspberry Pi

    * A PiPlates "DAQC2plate" is installed on the GPIO header. This provides the ADC needed to read intensity from the photodiode (via a Thorlabs photodiode amplifier)

    * USB connected to a ThorLabs MDT693B 3-axis piezo controller. The controller haves as a tty device via a serial console, generally at /dev/ttyACM0

    * Thorlabs NanoMax stage with at least open loop piezo actuators. This connects only to the above MDT693B.
        * We're using the NanoMax 300, which has a 20um range with input voltage from 0-75V.

    * An appropriate lens (e.g. Edmund high-precision lambda/40 aspheres are good) to focus the beam from free space onto the fiber

    * A photodiode that's monitoring power out of the fiber

    * A thorlabs photodiode "transimpedance amplifier" which the photodiode plugs into, and from which a BNC cable carrying an analog intensity signal from 0-10V is fed to analogue input pins on the PiPlate.


Installation and Pi confiuration:
    * Handy to have the ssh server enabled and vim installed

    * Install the "pyhidapi" library by doing "#apt install python3-hid"

    * Install the "pyserial" library by doing "pip3 install pyserial"

    * Clone this repo and make "main.py" executable
