# Initialize Variables
'''Only fill in for the first and third plane during initialization'''

x1, y1, z1, p1 = [] # First plane max point (y = 0.0)
x2, y2, z2, p2 = [] # Second plane max point (y = 37.5)
x3, y3, z3, p3 = [] # Third plane max point (y = 75.0)

radius = 37.5
error_margin = 1 # minimum radius size for search loop

# Calculate slopes
y2 = y1 + radius
x_slope = (x3 - x1) / 75
z_slope = (z3 - z1) / 75

# Find center plane
x = (x_slope * y1) + x1
z = (z_slope * y1) + z1
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
while radius > error_margin:

    radius /= 2 # shrink radius

    # Finding power and y-value of left-bound point (A)
    if y2 != 0:
        y1 = y2 - radius
        x = (x_slope * y1) + x1
        z = (z_slope * y1) + z1
        '''Find power value at (x, y1, z) and set as p1'''
    else:
        y1 = 0
        p1 = 0
    
    # Finding power and y-value of right-bound point (C)
    if y2 != 75:
        y3 = y2 + radius
        x = (x_slope * y3) + x1
        z = (z_slope * y3) + z1
        '''Find power value at (x, y3, z) and set as p3'''
    else:
        y3 = 0
        p3 = 0

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

