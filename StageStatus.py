import math
import sigfig
from typing import Union, Optional

from MovementClasses import StageDevices, Distance

VALID_AXES = ['x', 'y', 'z']


def run(stage: StageDevices, exposureTime: Union[int, float], which: Optional[str] = None,
        expose: bool = True, verbose: bool = True, log: bool = False):
    if which is not None:
        which = which.lower()
    else:
        which = 'all'

    assert which in ('all', 'general', 'stepper', 'piezo', 'sensor'), 'Invalid which input'

    if which == 'all':
        show_stepper = True
        show_piezo = True
        show_sensor = True
    else:
        show_stepper = False
        show_piezo = False
        show_sensor = False

    if which == 'stepper':
        show_stepper = True
    if which == 'piezo':
        show_piezo = True
    if which == 'sensor':
        show_sensor = True
    if which == 'general':
        show_piezo = True
        show_stepper = True

    lines = []
    lines.append(f"---------- {stage.name} status ----------\n")

    try:
        sensortype = stage.sensor.sensor.__class__.__name__
    except AttributeError:
        sensortype = stage.sensor.__class__.__name__    # for SimulationSensor
    if show_sensor:
        lines.append(f"Sensor type: {sensortype}")
        if sensortype == 'Socket':
            lines.append(f"Connection details:\n    host = {stage.sensor.sensor.host}")
            lines.append(f"    port = {stage.sensor.sensor.port}")
        if sensortype == 'Sipm' or sensortype == 'Photodiode':
            lines.append(f"Connection details:\n    address = {stage.sensor.sensor.addr}")
            lines.append(f"    channel = {stage.sensor.sensor.channel}")
        if sensortype == 'SimulationSensor':
            prop_axis = stage.sensor.propagation_axis
            deviation_plane = list(set(VALID_AXES).difference({prop_axis, }))
            deviation_plane = f"({deviation_plane[0]}, {deviation_plane[1]})"
            lines.append(f"Model details:\n    propagation axis = {prop_axis}")
            lines.append("    angle of deviation from propagation axis = " +
                    f"{sigfig.round(stage.sensor.angle * 180 / math.pi, 3, warn=False)} degrees," +
                         f" diagonal in the {deviation_plane} plane")
            lines.append(f"    peak intensity = {sigfig.round(stage.sensor.I0, 3, warn=False)}")
            waist_pos = str(stage.sensor.waist_pos) + ' microns'
            if verbose:
                lines.append(f"    waist = {Distance(stage.sensor.w0, 'microns').prettyprint()}")
            else:
                lines.append(f"    waist = {sigfig.round(stage.sensor.w0, 3, warn=False)} microns")
            if verbose:
                waist_pos = [Distance(pos, 'microns').prettyprint()
                                    for pos in stage.sensor.waist_pos]
                waist_pos = '\n' + ' '*28 + '( ' + (',\n' + ' '*30).join(waist_pos) + ' )'
            lines.append(f"    waist position (x, y, z) ={waist_pos}")
        if expose:
            lines.append(f"\nSensor current reading (exposure time {exposureTime}): " +
                         f"{sigfig.round(stage.sensor.integrate(exposureTime), 3, warn=False)}")
        lines.append('')
    if show_piezo:
        lines.append("Piezos:")
        if sensortype != 'SimulationSensor':
            if stage.axes['x'].piezo is None:
                lines.append("\nNo piezo controller connected")
                piezos_exist = False
            else:
                lines.append(f"    Port = {stage.piezo_port}\n    Baud rate = {stage.piezo_baud_rate}")
                piezos_exist = True
        else:
            piezos_exist = True
        for axis in VALID_AXES:
            if piezos_exist:
                if verbose:
                    lines.append(f"    {axis} = {stage.axes[axis].get_piezo_position().prettyprint()}")
                else:
                    lines.append(f"    {axis} = " +
                    f"{sigfig.round(stage.axes[axis].get_piezo_position().volts, 3, warn=False)} volts")
        lines.append('')
    if show_stepper:
        lines.append("Steppers:")
        for axis in VALID_AXES:
            if sensortype != 'SimulationSensor':
                if stage.axes[axis].stepper is None:
                    lines.append(f"    No {axis} stepper connected")
                    continue
                if not stage.axes[axis]._energized():
                    lines.append(f"    {axis} stepper (SN {stage.axes[axis].stepper_SN}) is not energized")
                    continue
                if stage.axes[axis]._position_uncertain():
                    lines.append(f"    {axis} (SN {stage.axes[axis].stepper_SN}) position is uncertain")
                    continue
            if verbose:
                lines.append(f"    {axis} (SN {stage.axes[axis].stepper_SN}) = " +
                             stage.axes[axis].get_stepper_position().prettyprint())
            else:
                lines.append(f"    {axis} (SN {stage.axes[axis].stepper_SN}) = " +
                         f"{sigfig.round(stage.axes[axis].get_stepper_position().microns, 3, warn=False)} microns")
        lines.append('')

    status = '\n'.join(lines)
    print(status)

    if log:
        log.info(status)

    return
