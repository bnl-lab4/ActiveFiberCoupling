from motion import move
from photodiode_in import get_exposure

# Main Function Code

def run(ser, file):

    # Initialize configurable variables

    length = 3 # Initial cube length of search volume.
    division = 5 # Number of sectors per length
    subplot_final = 5 # Total number of subplots to iterate

    # Initialize measurable variables
    
    ser.flush()
    ser.flushInput()
    ser.flushOutput()
    
    ser.write('xvoltage?\n'.encode()) # Query for x-voltage
    ser.readline().decode('utf-8').strip()
    x = ser.read(8).decode('utf-8').strip()
    x = x.replace("[","").replace("]","").replace(" ","").replace("<","")
    print(f"Response: {x}\n") # [OPTIONAL]
    max_x = float(x) - (length / 2) # x-coord of max power
    if max_x < 0: max_x = 0

    ser.write('yvoltage?\n'.encode()) # Query for y-voltage
    ser.readline().decode('utf-8').strip()
    y = ser.read(8).decode('utf-8').strip()
    y = y.replace("[","").replace("]","").replace(" ","")
    print(f"Response: {y}\n") # [OPTIONAL]
    max_y = float(y) - (length / 2) # y-coord of max power
    if max_y < 0: max_y = 0
    
    ser.write('zvoltage?\n'.encode()) # Query for z-voltage
    ser.readline().decode('utf-8').strip()
    z = ser.read(8).decode('utf-8').strip()
    z = z.replace("[","").replace("]", "").replace(" ","")
    print(f"Response: {z}\n") # [OPTIONAL]
    max_z = float(z) - (length / 2) # z-coord of max power
    if max_z < 0: max_z = 0
    
    ser.flush()
    
    # Initialize constants and lists

    subplot = 1 # Number of subplots iterated
    count = 0 # Number of movements [OPTIONAL]
    max_power = -1 # Maximum power intensity measured
    powers_list = [] # Full list of powers measured
    div_squared = division ** 2 # For index-related calculations

    # Commence Search
    while subplot <= subplot_final:

        # Calculate dimensional properties
        div_len = length / division # diameter of each sector
        mid = div_len / 2 # radius of each sector

        # Search all sectors in current subplot
        for i in list(range(division)):
            
            x_pos = max_x + (div_len * i) + mid # New x-position
            if x_pos >= 74: x_pos = 74
            move('x', x_pos, ser)

            for j in list(range(division)):
                
                y_pos = max_y + (div_len * j) + mid # New y-position
                if y_pos >= 74: y_pos = 0
                move('y', y_pos, ser)

                for k in list(range(division)):
                    
                    z_pos = max_z + (div_len * k) + mid # New z-position
                    if z_pos >= 74: z_pos = 74
                    move('z', z_pos, ser)
                    
                    # Measure and record power intensity
                    p_value = get_exposure(100)
                    powers_list.append(p_value)
                    if p_value > max_power:
                        max_power = p_value
                        index = (i * div_squared) + (j * division) + k
                    # Print current point count [OPTIONAL]
                    count += 1
                    print("Count: ", count)
        
        # Find coordinates of max power
        max_x += (div_len * (index // div_squared))
        max_y += (div_len * ((index % div_squared) // division))
        max_z += (div_len * ((index % div_squared) % division))

        # Update variables
        length /= division
        subplot += 1

    print(f"\nContinuous Search (Subplots 1-{subplot_final}):")
    file.write(f"\nContinuous Search (Subplots 1-{subplot_final}):\n")
    print(f"Powers: {powers_list}")
    file.write(f"Powers: {powers_list}\n")
    
    # Correct coordinates to center of sector
    max_x += mid
    max_y += mid
    max_z += mid

    # Move to final point
    move('x', max_x, ser)
    move('y', max_y, ser)
    move('z', max_z, ser)

    # Display and record final point
    print(f"New Point [Cont.]: ({max_x:.3f}, {max_y:.3f}, {max_z:.3f}, {max_power:.3f})\n")
    file.write(f"({max_x:.3f}, {max_y:.3f}, {max_z:.3f}, {max_power:.3f})\n")
    return