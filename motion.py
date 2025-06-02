def move(axis: str, position: float, ser):
    clamped = max(0.0, min(75.0, position))
    if clamped != position:
        print(f"Warning: requested {axis.upper()}={position:.2f}V, clamped to {clamped:.2f}V")
    command = f"{axis.lower()}voltage={clamped}\n"
    ser.write(command.encode())
    print(f"Moved {axis.upper()} to {clamped:.3f}V")
