from motion import move
from photodiode_in import get_exposure
import struct

# Main Function Code

def run(ser, file):

    # Initialize configurable variables

    length = 6 # Initial cube length of search volume.
    division = 3 # Number of sectors per length
    subplot_final = 10 # Total number of subplots to iterate

    # Initialize measurable variables
    ser.flush()
    ser.flushInput()
    ser.flushOutput()
    ser.write('xvoltage?\n'.encode()) # Query for x-voltage
    ser.readline().decode('utf-8').strip() # Receive output
    #x_bytes = ser.read(8)
    x = ser.read(8).decode('utf-8').strip()
    #x = ser.readline().decode('utf-8').strip()
    x = x.replace("[", "")
    x = x.replace("]", "")
    x = x.replace(" ", "")
    x = x.replace("<", "")
    #x = x.replace(
    #x = struct.unpack('>d', x_bytes)[0]
    print(f"Response: {x}\n") # [OPTIONAL]
    x = float(x)
    max_x = x - (length / 2) # x-coord of max power
    if max_x < 0: max_x = 0

    ser.write('yvoltage?\n'.encode()) # Query for y-voltage
    ser.readline().decode('utf-8').strip() # Receive output
    y = ser.read(8).decode('utf-8').strip()
    y = y.replace("[", "")
    y = y.replace("]", "")
    y = y.replace(" ", "")
    print(f"Response: {y}\n") # [OPTIONAL]
    y = float(y)
    max_y = y - (length / 2) # y-coord of max power
    if max_y < 0: max_y = 0
    
    ser.write('zvoltage?\n'.encode()) # Query for z-voltage
    ser.readline().decode('utf-8').strip() # Receive output
    z = ser.read(8).decode('utf-8').strip()
    z = z.replace("[", "")
    z = z.replace("]", "")
    z = z.replace(" ", "")
    z = float(z)
    print(f"Response: {z}\n") # [OPTIONAL]
    max_z = z - (length / 2) # z-coord of max power
    if max_z < 0: max_z = 0
    
    ser.flush()
    # Initialize constants and lists

    subplot = 1 # Number of subplots iterated
    count = 0 # Number of movements [OPTIONAL]

    x_voltages = [] # Full list of x-coords searched
    y_voltages = [] # Full list of y-coords searched
    z_voltages = [] # Full list of z-coords searched
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
                    
                    # Record current xyz-coordinates
                    x_voltages.append(x_pos)
                    y_voltages.append(y_pos)
                    z_voltages.append(z_pos)
                    # Measure and record power intensity
                    p_value = get_exposure(100) 
                    powers_list.append(p_value)

                    # Print current point count [OPTIONAL]
                    count += 1
                    print("Count: ", count)
        
        # Find max power and respective coordinates
        max_power = max(powers_list)
        max_index = powers_list.index(max_power)
        max_x = x_voltages[max_index] - mid
        max_y = y_voltages[max_index] - mid
        max_z = z_voltages[max_index] - mid

        '''# Print and write x-, z-, and p- values [OPTIONAL]
        print(f"\nSubplot {subplot}:\n")
        file.write(f"Current Subplot: {subplot}\n")
        print(x_voltages)
        file.write(f"X Voltages: {x_voltages}\n")
        print(y_voltages)
        file.write(f"Y Voltages: {y_voltages}\n")
        print(z_voltages)
        file.write(f"Z Voltages: {z_voltages}\n")
        print(powers_list)
        file.write(f"Power: {powers_list}\n")'''

        # Clear Lists
        x_voltages.clear()
        y_voltages.clear()
        x_voltages.clear()
        powers_list.clear()

        # Update range and increment subplot count
        length /= division
        subplot += 1
    
    # Display and record final result
    move('x', (max_x + mid), ser)
    move('y', (max_y + mid), ser)
    move('z', (max_z + mid), ser)
    print(f"\nNew Point: ({max_x + mid}, {max_y + mid}, {max_z + mid})")
    print(f"New Power: {max_power}\n")
    file.write(f"New Point and Power: ({max_x + mid}, {max_y + mid}, {max_z + mid}, {max_power})\n")
    #return
