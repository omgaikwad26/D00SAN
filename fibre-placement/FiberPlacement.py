from ultralytics import YOLO
import cv2
import cvzone
import math
import time
from scipy.optimize import fsolve
import socket
import numpy as np
import matplotlib.pyplot as plt
import math

#Camera Calibration         
def get_mouse_click(event, x, y, flags, param):
    global click_count
    global click_coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        click_coordinates.append((x,y))
        click_count +=1
        print(f'Click {click_count}: (x,y) = ({x}, {y})')
        if click_count == 4:
            cv2.setMouseCallback('Calibration Feed', lambda *args:None) #Disables mouse callback after 4 clicks

click_count = 0
click_coordinates = []

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cv2.namedWindow('Calibration Feed')
cv2.setMouseCallback('Calibration Feed', get_mouse_click)
#Camera Calibration, need 4 clicks
while True:
    ret, frame = cap.read()
    cv2.imshow('Calibration Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or click_count == 4:
        break

cap.release()
cv2.destroyAllWindows()

print(click_coordinates)                #the points displayed represent the calibration points

# Assigning the coordinates to the paper_coordinates dictionary
paper_coordinates = {
    'bottom_left': click_coordinates[0],
    'bottom_right': click_coordinates[1],
    'top_left': click_coordinates[2],
    'top_right': click_coordinates[3]
}

print (paper_coordinates)

# Function to generate path lines based on the angle
def generate_path_lines(angle, spacing=2):
    """
    Generate path lines for a given angle and spacing.
    """
    lines = []
    x_min, y_min = paper_coordinates['bottom_left']
    x_max, y_max = paper_coordinates['top_right']
    
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

    min_dim = min(x_start, y_start)
    max_dim = max(x_end, y_end)

    if angle == 0:

        y_coords = np.arange(y_start, y_end + spacing, spacing)

        for y in y_coords:
            print(angle, (x_start, y), (x_end , y))
            lines.append(((x_start, y), (x_end , y)))
            
    elif angle == 90:

        x_coords = np.arange(x_start, x_end + spacing, spacing)

        for x in x_coords:
            print(angle, (x, y_start), (x , y_end))
            lines.append(((x, y_start), (x , y_end)))

    else:
        #offset_range = np.arange(-max_dim, max_dim + spacing, spacing)
        offset_range = np.arange((10*max_dim), -(10*max_dim), -(spacing / sin_angle) )
        #offset_yrange = np.arange(min_dim, max_dim, spacing / cos_angle )

        for offset in offset_range:

            x0 = offset 
            y0 = y_start

            #Starting on bottom boundary
            if x0 >= x_start and x0 <= x_end:

                x1 = x_end
                y1 = ((x1 - x0) * tan_angle) + y0

                #bottom to top and right
                if x1 >= x_end and y1 >= y_start:

                    #bottom to right
                    if y1 <= y_end:
                        lines.append(((x0, y0), (x1, y1)))
                        print(angle, (x0, y0), (x1, y1))

                    #Bottom to Top
                    else:
                        y1 = y_end
                        x1 = ((y1 - y0) / tan_angle) + x0
                        lines.append(((x0, y0), (x1, y1)))
                        print(angle, (x0, y0), (x1, y1))
                
                #Bottom to Left and top
                elif y1 <= y_start:
                    x1 = x_start
                    y1 = -(((x0 - x1) * (tan_iota)) + y0)

                    #Bottom to Left 
                    if y1 <= y_end:
                        lines.append(((x0, y0), (x1, y1)))
                        #print (angle, (x1, y1))

                    #Bottom to top boundary
                    else:
                        y1 = y_end
                        x1 = (x0 - ((y0 - y1) / tan_iota))
                        #print (angle, (x1, y1))
                        lines.append(((x0, y0), (x1, y1)))

            #Starting on left boundary
            elif x0 < x_start:
                #print('left', angle, (x0, y0))

                y0 =((x_start - x0) * tan_angle) - y0
                x0 = x_start
                y1 = y_end
                x1 = ((y1 - y0) / tan_angle) + x0

                #Left to Right
                if x1 >= x_end:   
                    x1 = x_end
                    y1 = ((x1 - x0) * tan_angle) + y0
                    lines.append(((x0, y0), (x1, y1)))
                    print(angle, (x0, y0), (x1, y1))

                #if x0 >= y_min and x1 <= x_end and y_start >= y_start and y_end <= y_end
                #Left to Top
                elif y0 < y_end and y1 >= y_end and x1 >= x_start :
                    lines.append(((x0, y0), (x1, y1)))  
                    print(angle, (x0, y0), (x1, y1))

            # starting on right boundary
            elif x0 > x_end and y0 >= y_start:
                #print('right', angle, (x0, y0))

                y0 = - ((x0 - x_end) * tan_iota) + y_start
                x0 = x_end

                if y0 <= y_end:
                    x1 = x_start
                    y1 = -((x0 + x1) * tan_iota ) + y0

                    #print('right', angle, (x0, y0), (x1, y1))

                    #Right to top
                    if y1 >= y_end:
                        y1 = y_end
                        x1 = x0 + ((y1 - y0) / tan_iota)
                        lines.append(((x0, y0), (x1, y1)))
                        print (angle, (x0,y0), (x1,y1))

                    #Right to Left
                    elif y1 >= y_start:
                        lines.append(((x0, y0), (x1, y1)))
                        print (angle, (x0,y0), (x1,y1))
                

                #print (angle, (x0,y0))
            
    return lines

# Function to generate all paths for the fiber placement
def generate_fiber_paths(angle_repetitions, spacing=2):
    """
    Generate fiber placement paths for specified angles and repetitions.
    """
    paths = []
    
    for angle, repetitions in angle_repetitions:
        for _ in range(repetitions):
            paths.extend(generate_path_lines(angle, spacing))
    
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
    
    for edge in paper_edges:
        start, end = edge
        ax.plot([start[0], end[0]], [start[1], end[1]], 'k-')
    
    # Plot the fiber paths
    for path in paths:
        start, end = path
        ax.plot([start[0], end[0]], [start[1], end[1]], 'b-')
    
    ax.set_aspect('equal')
    ax.set_xlim(paper_coordinates['bottom_left'][0] - 1, paper_coordinates['bottom_right'][0] + 1)
    ax.set_ylim(paper_coordinates['bottom_left'][1] - 1, paper_coordinates['top_left'][1] + 1)
    plt.title("Fiber Placement Paths")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()

# Example usage
angle_repetitions = [(0, 1), (15, 1), (35, 1), (45, 1), (170, 1)]
spacing = 1
fiber_paths = generate_fiber_paths(angle_repetitions, spacing)

# Plot the generated paths
plot_fiber_paths(fiber_paths)
