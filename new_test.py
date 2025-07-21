# Main Function Code

def focal_search(start_plane, end_plane, error_radius = 1):

    # Initialize Variables
    x1, y1, z1, p1 = start_plane # First plane max point (y = 0.0)
    x3, y3, z3, p3 = end_plane # Third plane max point (y = 75.0)
    length = y3 - y1 # Range of planes to scan
    radius = length / 2 # Radius of search

    # Calculate slopes
    x_slope = (x3 - x1) / length
    z_slope = (z3 - z1) / length

    # Find center plane
    y2 = y1 + radius
    x = (x_slope * y2) + x1
    z = (z_slope * y2) + z1
    '''Find power value at (x, y2, z) and set as p2'''

    # Finding maximum power intensity
    y_values = [y1, y2, y3]
    power_values = [p1, p2, p3]
    max_power = max(power_values)
    max_index = power_values.index(max_power)
    # Set Center point values
    p2 = max_power # Set power for new center point (B)
    y2 = y_values[max_index] # set y-value for new center point (B)   

    # Search Loop
    while radius > error_radius:

        radius /= 2 # shrink radius
        y1, p1, y3, p3 = [0,0,0,0] # reset y- and p- values of bounds

        # Finding power and y-value of left-bound point (A)
        if y2 != 0:
            y1 = y2 - radius
            x = (x_slope * y1) + x1
            z = (z_slope * y1) + z1
            '''Find power value at (x, y1, z) and set as p1'''
                
        # Finding power and y-value of right-bound point (C)
        if y2 != length:
            y3 = y2 + radius
            x = (x_slope * y3) + x1
            z = (z_slope * y3) + z1
            '''Find power value at (x, y3, z) and set as p3'''

        # Finding maximum power intensity
        y_values = [y1, y2, y3]
        power_values = [p1, p2, p3]
        max_power = max(power_values)
        max_index = power_values.index(max_power)
        
        # Set Center point values
        p2 = max_power # Set power for new center point (B)
        y2 = y_values[max_index] # set y-value for new center point (B)   

    # Print final point
    x = (x_slope * y2) + x1
    z = (z_slope * y2) + z1
    Final_point = [x, y2, z, p2]
    return Final_point

