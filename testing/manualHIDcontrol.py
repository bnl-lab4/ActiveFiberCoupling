#!/usr/bin/python

import hid
import HIDvals as HIDval
from HIDvals import *


def main():
    # Open the Vaydeer multimedia volume controller knob.
    with hid.Device(ID_VENDOR, ID_PRODUCT) as dev:
        # Iterate until the user interrupts.
        while True:
            # Block until data is available, and then get the data.
            data = dev.read(SIZE_PACKET)

            # Check whether data was received, and check whether the data comes from a consumer control function.
            if data and data[IDX_REPORT] == ID_CONSUMER_CTRL:
                # Match the consumer control function.
                match data[IDX_FUNCTION]:
                    case HIDval.FN_VOLUME_DECREMENT:
                        print("Volume Decrement")

                    case HIDval.FN_VOLUME_INCREMENT:
                        print("Volume Increment")

                    case HIDval.FN_MUTE:
                        print("Mute")

                    case HIDval.FN_PLAY_PAUSE:
                        print("Play/Pause")

                    case HIDval.FN_EJECT:
                        print("Eject")

                    case HIDval.FN_STOP:
                        print("Stop")

                    case HIDval.FN_SCAN_PREVIOUS_TRACK:
                        print("Scan Previous Track")

                    case HIDval.FN_SCAN_NEXT_TRACK:
                        print("Scan Next Track")


if __name__ == "__main__":
    main()
