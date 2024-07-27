import struct
import os
import numpy as np
#import open3d as o3d
from scipy.optimize import fsolve
import time
import socket
import numpy as np
import matplotlib.pyplot as plt

# inverts the lagrange numerically


def forward_robot(x, *jtvecin):
    r, s = x
    rs = r * s
    jtvec = jtvecin[0]
    n1 = .25 * rs * (1-r) * (1-s)
    n2 = -.25 * rs * (1+r) * (1-s)
    n3 = .25 * rs * (1+r) * (1+s)
    n4 = -.25 * rs * (1-r) * (1+s)
    n5 = -.5 * (1-(r*r)) * s * (1-s)
    n6 = .5 * (1+r) * r * (1-(s*s))           #functions to get multiplication factor for rs and xy values
    n7 = .5 * (1-(r*r)) * s * (1+s)
    n8 = -.5 * (1-r) * r * (1-(s*s))
    n9 = (1-(r*r)) * (1-(s*s))
    jt1val = (jtvec[0][0] * n1 + jtvec[1][0] * n2 + jtvec[2][0] * n3 + jtvec[3][0] * n4
              + jtvec[4][0] * n5 + jtvec[5][0] * n6 + jtvec[6][0] * n7 + jtvec[7][0] * n8 + jtvec[8][0] * n9)
    jt2val = (jtvec[0][1] * n1 + jtvec[1][1] * n2 + jtvec[2][1] * n3 + jtvec[3][1] * n4
              + jtvec[4][1] * n5 + jtvec[5][1] * n6 + jtvec[6][1] * n7 + jtvec[7][1] * n8 + jtvec[8][1] * n9)
    jt3val = (jtvec[0][2] * n1 + jtvec[1][2] * n2 + jtvec[2][2] * n3 + jtvec[3][2] * n4
              + jtvec[4][2] * n5 + jtvec[5][2] * n6 + jtvec[6][2] * n7 + jtvec[7][2] * n8 + jtvec[8][2] * n9)
    jt4val = (jtvec[0][3] * n1 + jtvec[1][3] * n2 + jtvec[2][3] * n3 + jtvec[3][3] * n4
              + jtvec[4][3] * n5 + jtvec[5][3] * n6 + jtvec[6][3] * n7 + jtvec[7][3] * n8 + jtvec[8][3] * n9)
    jt5val = (jtvec[0][4] * n1 + jtvec[1][4] * n2 + jtvec[2][4] * n3 + jtvec[3][4] * n4
              + jtvec[4][4] * n5 + jtvec[5][4] * n6 + jtvec[6][4] * n7 + jtvec[7][4] * n8 + jtvec[8][4] * n9)
    jt6val = (jtvec[0][5] * n1 + jtvec[1][5] * n2 + jtvec[2][5] * n3 + jtvec[3][5] * n4
              + jtvec[4][5] * n5 + jtvec[5][5] * n6 + jtvec[6][5] * n7 + jtvec[7][5] * n8 + jtvec[8][5] * n9)  # Each joint angle calculation
    return [jt1val, jt2val, jt3val, jt4val, jt5val, jt6val]


def forwardlin(x, *ptvecin):
    # forward lagrange interpolation
    p, q = x
    ptvec = ptvecin[0]
    n1 = 0.25 * (1 - p) * (1 - q)
    n2 = 0.25 * (1 + p) * (1 - q)
    n3 = 0.25 * (1 + p) * (1 + q)
    n4 = 0.25 * (1 - p) * (1 + q)
    val1 = ptvec[0][0] * n1 + ptvec[1][0] * n2 + ptvec[2][0] * n3 + ptvec[3][0] * n4
    val2 = ptvec[0][1] * n1 + ptvec[1][1] * n2 + ptvec[2][1] * n3 + ptvec[3][1] * n4
    #print(p,q,val1,val2)
    return [val1, val2]

def forwardquad(x, *ptvecin):
    # forward lagrange interpolation
    r, s = x
    rs = r * s
    ptvec = ptvecin[0]
    n1 = .25 * rs * (1-r) * (1-s)
    n2 = -.25 * rs * (1+r) * (1-s)
    n3 = .25 * rs * (1+r) * (1+s)
    n4 = -.25 * rs * (1-r) * (1+s)
    n5 = -.5 * (1-(r*r)) * s * (1-s)
    n6 = .5 * (1+r) * r * (1-(s*s))
    n7 = .5 * (1-(r*r)) * s * (1+s)
    n8 = -.5 * (1-r) * r * (1-(s*s))
    n9 = (1-(r*r)) * (1-(s*s))
    val1 = (ptvec[0][0] * n1 + ptvec[1][0] * n2 + ptvec[2][0] * n3 + ptvec[3][0] * n4 + ptvec[4][0]
            * n5 + ptvec[5][0] * n6 + ptvec[6][0] * n7 + ptvec[7][0] * n8 + ptvec[8][0] * n9)
    val2 = (ptvec[0][1] * n1 + ptvec[1][1] * n2 + ptvec[2][1] * n3 + ptvec[3][1] * n4 + ptvec[4][1]
            * n5 + ptvec[5][1] * n6 + ptvec[6][1] * n7 + ptvec[7][1] * n8 + ptvec[8][1] * n9)
    return [val1, val2]


def reverse_error(x, *args):
    # error function for finding p,q roots
    rs = args[0][0]
    ptvec = args[0][1]
    x1 = forwardquad(x, ptvec)
    err = [(x1[0] - rs[0]) * (x1[0] - rs[0]), (x1[1] - rs[1]) * (x1[1] - rs[1])]
    return err


def reverse(r, s, *ptvec):
    data = [[r, s], ptvec[0]]
    x0 = [0.25, 0.25]
    p = fsolve(reverse_error, x0, data)
    return p


def map(r, s, rspts, xypts, jtpts):
    p = reverse(r, s, rspts)
    x = forwardquad(p, xypts)
    j = forward_robot(p, jtpts)
    return x, j

"""
1 Starts at bottom left and goes around the corners, 
5 is in the middle of the bottom side and goes around the sides,
9 is center
Coordinates are in millimeters
"""
"""
Paper Coordinates
pt 1: (685, 210)
pt 2:
pt 3: 
pt 4: 
pt 5: 
pt 6: 
pt 7: 
pt 8: 
pt 9:  


"""
#Mapping Coordinates in mm
#ptvec = [[a,b] for a, b in click_coordinates]
ptvec = (
    [0, 0],             #1
    [107.95, 0],           #2
    [215.9, 0],         #3
    [215.9, 139.7],           #4
    [215.9, 279.4],           #5
    [107.95, 279.4],         #6
    [0, 279.4],         #7
    [0, 139.7],           #8
    [202, 202]          #9
)

# Actual Coordinates
xyvec = (
    [0, 0],             #1
    [107.95, 0],           #2
    [215.9, 0],         #3
    [215.9, 139.7],           #4
    [215.9, 279.4],           #5
    [107.95, 279.4],         #6
    [0, 279.4],         #7
    [0, 139.7],           #8
    [202, 202] 
)

# Joint Angles
jtvec = (
    [81.94, -37.56, -114.07, 0, -25.4, 0],
    [90.07,-39.18,-111.22,0,-32.17,0.77],
    [97.52, -38.03, -113.77, 0, -25.61, 8.96],
    [96.46, -44.26, -98.66, 0, -36.43, 6.63],
    [95.72, -50.83, -82.93, 0, -44.40, 8.96],
    [90.29, -51.16, -83.04, 0, -44.48, 6.63],
    [84.09, -51.30, -82.66, 0,-44.40,-6.42],
    [83.02, -46.13, -93.33, 0, -44.39, -6.42],
    [89.76, -44.34, -97.32, 0, -39.27, -1.23]
)

# Assigning the coordinates to the paper_coordinates dictionary
paper_coordinates = {
    'bottom_left': (215.9, 0),
    'bottom_right': (0, 0),
    'top_left':  (215.9, 279.4),
    'top_right': (0, 279.4)
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
def placeFiber(paths):
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
    ax.set_xlim(paper_coordinates['bottom_left'][0] + 1, paper_coordinates['bottom_right'][0] - 1)
    ax.set_ylim(paper_coordinates['bottom_right'][1] - 1, paper_coordinates['top_left'][1] + 1)

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
startxy, endxy = placeFiber(fiber_paths)

numberOfLines = len(fiber_paths)

startjtvec = []
endxyvec = []
endjtvec = [] 

for point in startxy:
    x, j = map(point[0], point[1], ptvec, xyvec, jtvec)
    startjtvec.append([round(num, 2) for num in j])

for point in endxy:
    x, j = map(point[0], point[1], ptvec, xyvec, jtvec)
    endxyvec.append([round(num, 2) for num in x])
    endjtvec.append([round(num, 2) for num in j])

for i in range(numberOfLines):

    print (startjtvec[i], endjtvec[i])


