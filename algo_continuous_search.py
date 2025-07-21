from motion import move
from photodiode_in import get_exposure

# Main Function Code

def run(ser, file):
    
    command = f"xvoltage?\n"
    ser.read(ser.in_waiting).decode("utf-8")
    ser.write(command.encode())
    ser.read(ser.in_waiting).decode("utf-8")
    x = ser.readline().decode().strip()
    #ser.flush()
    print(f"X currently at {x}")
'''
    length = 75 # Initial cube length of search volume.
    # Initialize variables (x, y, z = original point)
    max_x = x - (length / 2) # x-coord of max power
    if max_x < 0: max_x = 0
    max_y = y - (length / 2) # y-coord of max power
    if max_y < 0: max_y = 0
    max_z = z - (length / 2)# z-coord of max power
    if max_z < 0: max_z = 0

    count = 0 # Number of movements [OPTIONAL]
    subplot = 1 # Number of subplots iterated
    division = 3 # Number of sectors per length
    subplot_final = 10 # Total number of subplots to iterate

    x_voltages = [] # Full list of x-coords searched
    y_voltages = [] # Full list of y-coords searched
    z_voltages = [] # Full list of z-coords searched
    powers_list = [] # Full list of powers measured

    while subplot <= subplot_final:

        # Calculate quadrant coordinates
        div_len = length / division # diameter of each sector
        mid = div_len / 2 # radius of each sector

        # Search through quadrants
        for i in list(range(division)):
            
            x_pos = max_x + (div_len * i) + mid
            if x_pos < 74: move('x', x_pos, ser)
            else: move('x', 74, ser)
            
            for j in list(range(division)):
                
                y_pos = max_y + (div_len * j) + mid
                if y_pos < 74: move('y', y_pos, ser)
                else: move('y', 74, ser)
                
                for k in list(range(division)):
                    
                    z_pos = max_z + (div_len * k) + mid
                    if z_pos < 74: move('z', z_pos, ser)
                    else: move('z', 74, ser)
                    
                    x_voltages.append((div_len * i + mid))
                    y_voltages.append((div_len * j + mid))
                    z_voltages.append((div_len * k + mid))
                    
                    p_value = get_exposure(100) 
                    powers_list.append(p_value)

                    # Print current point count [OPTIONAL]
                    count += 1
                    print("Count: ", count)
        
        # Find max power and respective coordinates
        max_power = max(powers_list)
        max_index = powers_list.index(max_power)
        max_x = x_voltages(max_index) - mid
        max_y = y_voltages(max_index) - mid
        max_z = z_voltages(max_index) - mid

        # Print and write x-, z-, and p- values [OPTIONAL]
        print("\nSubplot 1:\n")
        file.write(f"Current Subplot: {subplot}\n")
        print(x_voltages)
        file.write(f"X Voltages: {x_voltages}\n")
        print(y_voltages)
        file.write(f"Y Voltages: {y_voltages}\n")
        print(z_voltages)
        file.write(f"Z Voltages: {z_voltages}\n")
        print(powers_list)
        file.write(f"Power: {powers_list}\n")

        # Clear Lists
        x_voltages.clear()
        y_voltages.clear()
        x_voltages.clear()
        powers_list.clear()

        # Update range and increment subplot count
        length /= division
        subplot += 1

    #return max_x, max_y, max_z, max_power
'''
