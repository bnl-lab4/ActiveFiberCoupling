"""
Reports the status of a stage (`movement_classes.StageDevices`).
"""

import math
from typing import Optional

import sigfig

from logging_utils import get_logger
from movement_classes import Distance, StageDevices

# unique logger name for this module
logger = get_logger(__name__)

VALID_AXES = ["x", "y", "z"]


def run(
    stage: StageDevices,
    exposure_time: int | float,
    which: Optional[str] = None,
    expose: bool = True,
    verbose: bool = True,
    log: bool = False,
) -> None:
    """
    Print the status of all/some of steppers, piezos, and sensor.

    Parameters
    ----------
    stage : `movement_classes.StageDevices`
        Stage to report the status of.
    exposure_time : int, float
        Exposure time to use for reporting sensor reading.
    which : {'all', 'general', 'stepper', 'piezo', 'sensor', None}, optional
        Which parts of the stage to report the status of. 'all' and
        ``None`` (default) report on the steppers, piezos, and sensor.
        'general' reports on the steppers and piezos, but not the sensor.
    expose : bool, default=True
        Whether to take and report a reading from the sensor.
    verbose : bool, default=True
        Whether to report values stored as `Distance.Distance` objects
        in all units (using ``prettyprint()``) or just in microns.
    log : bool, default=False
        Whether to log the report.

    Returns
    -------
    str
        Multiline report on stage status.
    """

    if which is None:
        which = "all"
    else:
        which = which.lower()

    if which not in ("all", "general", "stepper", "piezo", "sensor"):
        raise ValueError("Invalid input for which")

    if which == "all":
        show_stepper = True
        show_piezo = True
        show_sensor = True
    else:
        show_stepper = False
        show_piezo = False
        show_sensor = False

    if which == "stepper" or which == "general":
        show_stepper = True
    if which == "piezo" or which == "general":
        show_piezo = True
    if which == "sensor":
        show_sensor = True

    lines = []
    lines.append(f"---------- {stage.name} status ----------\n")

    try:
        sensortype = stage.sensor.sensor.__class__.__name__
    except AttributeError:
        sensortype = stage.sensor.__class__.__name__  # for SimulationSensor
    if show_sensor:
        lines.append(f"Sensor type: {sensortype}")
        if sensortype == "Socket":
            lines.append(f"Connection details:\n    host = {stage.sensor.sensor.host}")
            lines.append(f"    port = {stage.sensor.sensor.port}")
        if sensortype == "Piplate":
            lines.append(
                f"Connection details:\n    address = {stage.sensor.sensor.addr}"
            )
            lines.append(f"    channel = {stage.sensor.sensor.channel}")
        if sensortype == "SimulationSensor":
            prop_axis = stage.sensor.propagation_axis
            deviation_plane = list(
                set(VALID_AXES).difference(
                    {
                        prop_axis,
                    }
                )
            )
            deviation_plane = f"({deviation_plane[0]}, {deviation_plane[1]})"
            lines.append(f"Model details:\n    propagation axis = {prop_axis}")
            lines.append(
                "    angle of deviation from propagation axis = "
                + f"{sigfig.round(stage.sensor.angle * 180 / math.pi, 3, warn=False)} degrees,"
                + f" diagonal in the {deviation_plane} plane"
            )
            lines.append(
                f"    peak intensity = {sigfig.round(stage.sensor.I0, 3, warn=False)}"
            )
            waist_pos = str(stage.sensor.waist_pos) + " microns"
            if verbose:
                lines.append(
                    f"    waist = {Distance(stage.sensor.w0, 'microns').prettyprint()}"
                )
            else:
                lines.append(
                    f"    waist = {sigfig.round(stage.sensor.w0, 3, warn=False)} microns"
                )
            if verbose:
                waist_pos = [
                    Distance(pos, "microns").prettyprint()
                    for pos in stage.sensor.waist_pos
                ]
                waist_pos = (
                    "\n" + " " * 28 + "( " + (",\n" + " " * 30).join(waist_pos) + " )"
                )
            lines.append(f"    waist position (x, y, z) ={waist_pos}")
        if expose:
            lines.append(
                f"\nSensor current reading (exposure time {exposure_time}): "
                + f"{sigfig.round(stage.sensor.integrate(exposure_time), 3, warn=False)}"
            )
        lines.append("")
    if show_piezo:
        lines.append("Piezos:")
        if sensortype != "SimulationSensor":
            if stage.axes["x"].piezo is None:
                lines.append("\nNo piezo controller connected")
                piezos_exist = False
            else:
                lines.append(
                    f"    Port = {stage.piezo_port}\n    Baud rate = {stage.piezo_baud_rate}"
                )
                piezos_exist = True
        else:
            piezos_exist = True
        for axis in VALID_AXES:
            if piezos_exist:
                if verbose:
                    lines.append(
                        f"    {axis} = {stage.axes[axis].get_piezo_position().prettyprint()}"
                    )
                else:
                    lines.append(
                        f"    {axis} = "
                        + f"{sigfig.round(stage.axes[axis].get_piezo_position().volts, 3, warn=False)} volts"
                    )
        lines.append("")
    if show_stepper:
        lines.append("Steppers:")
        for axis in VALID_AXES:
            if sensortype != "SimulationSensor":
                if stage.axes[axis].stepper is None:
                    lines.append(f"    No {axis} stepper connected")
                    continue
                if not stage.axes[axis]._energized():
                    lines.append(
                        f"    {axis} stepper (SN {stage.axes[axis].stepper_sn}) is not energized"
                    )
                    continue
                if stage.axes[axis]._position_uncertain():
                    lines.append(
                        f"    {axis} (SN {stage.axes[axis].stepper_sn}) position is uncertain"
                    )
                    continue
            if verbose:
                lines.append(
                    f"    {axis} (SN {stage.axes[axis].stepper_sn}) = "
                    + stage.axes[axis].get_stepper_position().prettyprint()
                )
            else:
                lines.append(
                    f"    {axis} (SN {stage.axes[axis].stepper_sn}) = "
                    + f"{sigfig.round(stage.axes[axis].get_stepper_position().microns, 3, warn=False)} microns"
                )
        lines.append("")

    status = "\n".join(lines)
    print(status)

    if log:
        logger.info(status)

    return
