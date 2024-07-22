#from ultralytics import YOLO
import cv2
import cvzone
import math
import time
from scipy.optimize import fsolve
import socket
import numpy as np
import matplotlib.pyplot as plt

arena_coordinates = {
    'bottom_left': (945, 129),
    'bottom_right': (481, 119),
    'top_left':  (1187, 683),
    'top_right': (110, 574)
}

# Assigning the coordinates to the paper_coordinates dictionary
paper_coordinates = {
    'bottom_left': (760, 220),
    'bottom_right': (580, 220),
    'top_left':  (760, 380),
    'top_right': (580, 380)
}

print (paper_coordinates)

# Function to generate path lines based on the angle
def generate_path_lines(angle, spacing=20):
    """
    Generate path lines for a given angle and spacing.
    """
    lines = []
    #x_min, y_min = paper_coordinates['bottom_left']
    #x_max, y_max = paper_coordinates['top_right']

    x_min, y_min = paper_coordinates['bottom_right']
    x_max, y_max = paper_coordinates['top_left']
    
    angle_rad = np.deg2rad(angle)
    tan_angle = np.tan(angle_rad)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    #for angles > 90
    iota_rad = np.deg2rad(180 - angle)
    tan_iota = - np.tan(iota_rad)

    x_start = x_min 
    y_start = y_min 
    x_end = x_max 
    y_end = y_max 

    print ((x_start, y_start), (x_end, y_end))

    min_dim = min(x_start, y_start)
    max_dim = max(x_end, y_end)

    if angle == 0:

        y_coords = np.arange(y_start, y_end, spacing)

        for y in y_coords:
            print(angle, (x_start, y), (x_end , y))
            lines.append(((x_start, y), (x_end , y)))
            
    if angle == 90:
        #print (angle)

        x_coords = np.arange(x_start, x_end, spacing)
        #print (angle, x_coords)

        for x in x_coords:
            print(angle, (x, y_start), (x , y_end))
            lines.append(((x, y_start), (x , y_end)))

    else:
        #offset_range = np.arange(-max_dim, max_dim + spacing, spacing)
        offset_range = np.arange(-(10*max_dim), (10*max_dim), (spacing / sin_angle) )
        #print(offset_range)
        #offset_yrange = np.arange(min_dim, max_dim, spacing / cos_angle )

        for offset in offset_range:

            x0 = offset 
            y0 = y_start

            #Starting on bottom boundary
            if x0 >= x_start and x0 <= x_end:

                x1 = x_end
                y1 = ((x1 - x0) * tan_angle) + y0

                #bottom to top and left
                if x1 >= x_end and y1 >= y_start:

                    #bottom to left
                    if y1 <= y_end:
                        lines.append(((x0, y0), (x1, y1)))
                        print("B-L",angle, (x0, y0), (x1, y1))

                    #Bottom to Top
                    else:
                        y1 = y_end
                        x1 = ((y1 - y0) / tan_angle) + x0
                        lines.append(((x0, y0), (x1, y1)))
                        print("B-LT",angle, (x0, y0), (x1, y1))
                
                #Bottom to Right and top
                elif y1 <= y_start:
                    x1 = x_start
                    y1 = (((x1 - x0) * (tan_iota)) + y0)

                    #Bottom to Right 
                    if y1 <= y_end:
                        lines.append(((x0, y0), (x1, y1)))
                        print ("B-R",angle, (x1, y1))

                    #Bottom to top boundary
                    else:
                        y1 = y_end
                        x1 = (x0 - ((y0 - y1) / tan_iota))
                        print("B-RT",angle, (x1, y1))
                        lines.append(((x0, y0), (x1, y1)))
            
            #Starting on right boundary
            elif x0 < x_start:
                #print('right', angle, (x0, y0))

                y0 =((x_start - x0) * tan_angle) + y0
                x0 = x_start
                y1 = y_end
                x1 = ((y1 - y0) / tan_angle) + x0

                #Right to Left
                if x1 >= x_end:   
                    x1 = x_end
                    y1 = ((x1 - x0) * tan_angle) + y0
                    lines.append(((x0, y0), (x1, y1)))
                    print("R-L",angle, (x0, y0), (x1, y1))
                
                #if x0 >= y_min and x1 <= x_end and y_start >= y_start and y_end <= y_end
                #Right to Top
                elif y0 >= y_start and y0 <= y_end:
                    lines.append(((x0, y0), (x1, y1)))  
                    print("R-T",angle, (x0, y0), (x1, y1))
                
            # starting on left boundary
            elif x0 > x_end:
                #print('left', angle, (x0, y0))

                y0 = - ((x0 - x_end) * tan_iota) + y_start
                x0 = x_end
                x1 = x_start
                y1 = ((x1 -x0) * tan_iota ) + y0

                #print('left', angle, (x0, y0), (x1, y1))
                #Left to Right
                if y1 <= y_end and y0 >= y_start:
                    lines.append(((x0, y0), (x1, y1)))
                    print ("L-R", angle, (x0,y0), (x1,y1))

                
                #Left to top
                elif y1 >= y_end and y0 <= y_end:
                    y1 = y_end
                    x1 = x0 + ((y1 - y0) / tan_iota)
                    lines.append(((x0, y0), (x1, y1)))
                    print ("L-T", angle, (x0,y0), (x1,y1))
            
    return lines

# Function to generate all paths for the fiber placement
def generate_fiber_paths(angle_repetitions, spacing=30):
    """
    Generate fiber placement paths for specified angles and repetitions.
    """
    paths = []
    
    for angle, repetitions in angle_repetitions:
        for _ in range(repetitions):
            paths.extend(generate_path_lines(angle, spacing))
    
    print ("Paths", paths)
    return paths

# Function to plot the paths
def plot_fiber_paths(paths):
    """
    Plot the fiber paths on the paper.
    """
    fig, ax = plt.subplots()
    
    # Plot the paper boundaries
    paper_edges = [
        (paper_coordinates['bottom_left'], paper_coordinates['bottom_right']),
        (paper_coordinates['bottom_right'], paper_coordinates['top_right']),
        (paper_coordinates['top_right'], paper_coordinates['top_left']),
        (paper_coordinates['top_left'], paper_coordinates['bottom_left']),
    ]
    """
    for edge in paper_edges:
        start, end = edge
        ax.plot([start[0], end[0]], [start[1], end[1]], 'k-')
    """
    """
    # Plot the fiber paths
    for path in paths:
        start, end = path
        ax.plot([start[0], end[0]], [start[1], end[1]], 'b-')
    """
    # Plot the fiber paths
    for i, path in enumerate(paths):
        start, end = path
        if i % 2 != 0:
            start, end = end, start
            color = 'g-'
        else:
            color = 'b-'

        ax.plot([start[0],end[0]], [start[1],end[1]], color)
        print ("path", start, end)
    
    ax.set_aspect('equal')
    ax.set_xlim(arena_coordinates['bottom_left'][0] + 1, arena_coordinates['bottom_right'][0] - 1)
    ax.set_ylim(arena_coordinates['bottom_right'][1] - 1, arena_coordinates['top_left'][1] + 1)

    plt.title("Fiber Placement Paths")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    plt.grid(True)
    plt.show()


# Example usage
angle_repetitions = [(0, 1), (20, 1), (160, 1), (90, 1)]
spacing = 30
fiber_paths = generate_fiber_paths(angle_repetitions, spacing)

numberOfLines = len(fiber_paths)
print("number of lines", numberOfLines)

# Plot the generated paths
plot_fiber_paths(fiber_paths)
