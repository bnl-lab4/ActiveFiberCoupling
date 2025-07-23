from motion import move
from photodiode_in import get_exposure

# Main Function Code

def run(ser, file):

    # Initialize configurable variables

    length = 3 # Initial cube length of search volume.
    division = 5 # Number of sectors per length
    subplot_final = 5 # Total number of subplots to iterate

    # Initialize measurable variables

    # Initialize measurable variables
    
    ser.flush()
    ser.flushInput()
    ser.flushOutput()
    
    ser.write('xvoltage?\n'.encode()) # Query for x-voltage
    ser.readline().decode('utf-8').strip() # Receive output
    x = ser.read(8).decode('utf-8').strip()
    x = x.replace("[","").replace("]","").replace(" ","").replace("<","")
    print(f"Response: {x}\n") # [OPTIONAL]
    max_x = float(x) - (length / 2) # x-coord of max power
    if max_x < 0: max_x = 0

    ser.write('yvoltage?\n'.encode()) # Query for y-voltage
    ser.readline().decode('utf-8').strip() # Receive output
    y = ser.read(8).decode('utf-8').strip()
    y = y.replace("[","").replace("]","").replace(" ","")
    print(f"Response: {y}\n") # [OPTIONAL]
    max_y = float(y) - (length / 2) # y-coord of max power
    if max_y < 0: max_y = 0
    
    ser.write('zvoltage?\n'.encode()) # Query for z-voltage
    ser.readline().decode('utf-8').strip() # Receive output
    z = ser.read(8).decode('utf-8').strip()
    z = z.replace("[","").replace("]", "").replace(" ","")
    print(f"Response: {z}\n") # [OPTIONAL]
    max_z = float(z) - (length / 2) # z-coord of max power
    if max_z < 0: max_z = 0
    
    ser.flush()
    
    # Initialize constants and lists

    subplot = 1 # Number of subplots iterated
    count = 0 # Number of movements [OPTIONAL]
    powers_list = [] # Full list of powers measured

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
                    # Print current point count [OPTIONAL]
                    count += 1
                    print("Count: ", count)
        
        # Find max power and respective coordinates
        max_power = max(powers_list)
        index = powers_list.index(max_power)
        
        div_squared = division ** 2
        max_x += (div_len * (index // (div_squared)))
        max_y += (div_len * ((index % div_squared) // division))
        max_z += (div_len * ((index % div_squared) % division))

        # Update variables; clear list
        length /= division
        subplot += 1
        powers_list.clear()
    
    # Move to new point
    move('x', max_x + mid, ser)
    move('y', max_y + mid, ser)
    move('z', max_z + mid, ser)

    # Display and record final result
    print(f"\nNew Point: ({max_x + mid}, {max_y + mid}, {max_z + mid}, {max_power})")
    print(f"Power: {max_power}\n")
    file.write(f"New point and power({max_x + mid}, {max_y + mid}, {max_z + mid}, {max_power})\n")

    #return
