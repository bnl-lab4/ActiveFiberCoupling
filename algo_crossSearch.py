from motion import move
from photodiode_in import getPower
import time


def xscan(step, start, stop, ser):
    voltages = []
    powers = []
    position = start
    while position <= stop:
        move('x', position, ser)
        power = getPower(200)
        print(f"X {position: .2f} V, Power = {power: .6f}")
        voltages.append(position)
        powers.append(power)
        position += step

    index_localpeak = powers.index(max(powers))
    voltage_localpeak = voltages[index_localpeak]
    print(f"\nLocal peak = {powers[index_localpeak]: .6f} at X = {voltage_localpeak: .2f} V")
    move('x', voltage_localpeak, ser)
    time.sleep(0.075)
    return voltage_localpeak

def zscan(step, start, stop, ser):
    voltages = []
    powers = []
    position = start
    while position <= stop:
        move('z', position, ser)
        power = getPower(200)
        print(f"Z {position: .2f} V, Power = {power: .6f}")
        voltages.append(position)
        powers.append(power)
        position += step

    index_localpeak = powers.index(max(powers))
    voltage_localpeak = voltages[index_localpeak]
    print(f"\nLocal peak = {powers[index_localpeak]: .6f} at X = {voltage_localpeak: .2f} V")
    move('z', voltage_localpeak, ser)
    time.sleep(0.075)
    return voltage_localpeak


def run(ser):
    print("Running cross search algorithm...")
    step = 1
    start = 0
    stop = 75
    #focVoltages = []
    #focPowers = []

    xpeak = xscan(step, start, stop, ser)
    zpeak = zscan(step, start, stop, ser)

    step = 0.5
    start = xpeak - 5
    stop = xpeak + 5

    xpeak = xscan(step, start, stop, ser)

    start = zpeak - 8
    stop = zpeak + 8

    zpeak = zscan(step, start, stop, ser)





    print("Cross Search complete.")
