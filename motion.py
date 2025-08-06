import time

def move(axis: str, position: float, ser, ch, file):
    clamped = max(0.0, min(75.0, position))
    if clamped != position:
        print(f"Warning: requested {axis.upper()}={position:.2f}V, clamped to {clamped:.2f}V")
    command = f"{axis.lower()}voltage={clamped}\n"
    #print("serial says: ", ser.read(ser.in_waiting).decode("utf-8"))
    ser.read(ser.in_waiting).decode("utf-8")
    ser.write(command.encode())
    ser.flush()
    timestamp = str(time.perf_counter())
    file.write(f"{timestamp}: ")
    file.write(f"{axis.upper()}{ch} {clamped:.3f}V\n")

    return clamped
