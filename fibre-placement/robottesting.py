from ultralytics import YOLO
import cv2
import cvzone
import math
import time
from scipy.optimize import fsolve
import socket
import numpy as np
import matplotlib.pyplot as plt

# Global variables for calibration
click_count = 0
click_coordinates = []

def get_mouse_click(event, x, y, flags, param):
    global click_count
    global click_coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        click_coordinates.append((x, y))
        click_count += 1
        print(f'Click {click_count}: (x, y) = ({x}, {y})')
        if click_count == 9:
            cv2.setMouseCallback('Corners Marked', lambda *args: None)  # Disables mouse callback after 4 clicks

# Capture the video feed
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cv2.namedWindow('Çalibration')
cv2.setMouseCallBack('Calibration Feed', get_mouse_click)
# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Failed to capture image")
    cap.release()
    exit()



print(click_coordinates)  # the points displayed represent the calibration points

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


fileName = input("Enter file name: ")
coordinates, size = SortingCoordinates(fileName)

"""
1 Starts at bottom left and goes around the corners, 
5 is in the middle of the bottom side and goes around the sides,
9 is center
Coordinates are in millimeters
"""
"""
Paper Coordinates
pt 1: [653,177]
pt 2: [733, 185]
pt 3: [809,189]
pt 4: [795, 262]
pt 5: [784,345]
pt 6: [694, 333]
pt 7: [599,323]
pt 8: [627, 246]
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
    [0, 0],
    [404, 0],
    [404, 404],
    [0, 404],
    [202, 0],
    [404, 202],
    [202, 404],
    [0, 202],
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

coords = map(ptvec, xyvec, jtvec)
print (coords)


"""
jtanglesStart = coords[1]
jtanglesEnd = coords[2]
#server is opened which connects directly to the robot 
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = "192.168.137.50"
port = 8085
server_socket.bind((host, port))

server_socket.listen(5)
client_socket, addr = server_socket.accept()

def received_message():
    recieved_data = client_socket.recv(1024)
    print(recieved_data)

"""
Robot goes to the first start position
moves down
draws the first line
moves up
Repeat 1-4
"""

moveToStart = [b"moveto", jtanglesStart]
moveToEnd = [b"moveto", jtanglesEnd]
closeGripper = b"close gripper"
openGripper = b"open gripper"
moveupvar = b"40" #move up variable
movedownvar= b"-25" #move down variable
moveup = [b"moveup", moveupvar]
movedown = [b"movedown", movedownvar]
rotatevar = b"90"
rotategripper = [b"rotate gripper", rotatevar]

def sendmessage(message):
    client_socket.send(b"start")
    time.sleep(0.7)
    if message[0] == b'moveto':
        client_socket.send(message[0])
        time.sleep(0.7)
        #received_message() # Small delay to ensure separation
        
        for jtangles in message[1]:
            time.sleep(0.7)
            client_socket.send(bytes(str(jtangles), 'utf-8'))
            time.sleep(0.7)
            #received_message() # Small delay to ensure separation
        client_socket.send(b'end')
        time.sleep(0.7)
    elif message[0] == b'moveup' or message[0] == b'movedown':
        client_socket.send(message[0])
        time.sleep(0.7)
        client_socket.send(message[1])
        time.sleep(0.7)
        client_socket.send(b'end')
    elif message[0] == b'rotate gripper':
        client_socket.send(message[0])
        time.sleep(.7)
        client_socket.send(message[1])
        time.sleep(.7)
        client_socket.send(b'end')
    else:
        client_socket.send(message)
        time.sleep(0.7)
        client_socket.send(b'end')
        time.sleep(0.7)
    return 0



sendmessage(closeGripper)
received_message()
sendmessage(moveToStart) 
received_message()
sendmessage(movedown)
received_message()
sendmessage(moveToEnd)
received_message()
sendmessage(moveup)
received_message()
"""



import struct
import os
import numpy as np
#import open3d as o3d
from scipy.optimize import fsolve
import time
import socket


def read_vox_file(file_path):
    with open(file_path, 'rb') as f:
        # Read header
        header = f.read(4)
        if header != b'VOX ':
            raise ValueError("Not a valid .vox file")

        version = struct.unpack('<I', f.read(4))[0]

        if version != 200:
            raise ValueError("Not a valid .vox file")

        # Read chunks
        while f.tell() < os.fstat(f.fileno()).st_size:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break
            chunk_size = struct.unpack('<I', f.read(4))[0]
            children_size = struct.unpack('<I', f.read(4))[0]
            chunk_data = f.read(chunk_size)

            if chunk_id == b'SIZE':
                size_x, size_y, size_z = struct.unpack('<III', chunk_data[:12])
            elif chunk_id == b'XYZI':
                num_voxels = struct.unpack('<I', chunk_data[:4])[0]
                voxels = struct.unpack('<' + 'BBBB' * num_voxels, chunk_data[4:])
                voxels = [(voxels[i], voxels[i + 1], voxels[i + 2], voxels[i + 3]) for i in range(0, len(voxels), 4)]

        return {
            'size': (size_x, size_y, size_z),
            'voxels': voxels
        }


"""
# Creates Visualization of the Voxel Image
def visualize_voxels(points, voxel_size):
    pcd_ = o3d.geometry.PointCloud()
    pcd_.points = o3d.utility.Vector3dVector(points)
    voxelGrid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_, voxel_size)
    o3d.visualization.draw_geometries([voxelGrid])
"""


# Adds a Layer Number for each row to sort individual layers
def layeringRows(arr, column):
    layeredArr = []
    prevRow = arr[0]
    layer = 0
    rows = 0
    layers = [0]
    for row in arr:
        if row[column] == prevRow[column]:
            row_with_layer = list(row) + [layer]
            rows += 1
        else:
            layer += 1
            layers.append(rows)
            rows += 1
            row_with_layer = list(row) + [layer]
        layeredArr.append(row_with_layer)
        prevRow = row
    layers.append(rows)
    return layeredArr, layers


# Creates a partition for quicksort
def partition(array, low, high):
    pivot = array[high]
    i = low - 1
    for j in range(low, high):
        if array[j] <= pivot:
            i = i + 1
            (array[i], array[j]) = (array[j], array[i])
    (array[i + 1], array[high]) = (array[high], array[i + 1])
    return i + 1


# Quicksort Algorithm
def quickSort(array, low, high):
    if low < high:
        pi = partition(array, low, high)
        quickSort(array, low, pi - 1)
        quickSort(array, pi + 1, high)


# Final Sort Algorithm to sort each axis
def sortEachLayer(arr, ranges):
    for i in range(len(ranges) - 1):
        quickSort(arr, ranges[i], ranges[i + 1] - 1)
    return arr


# Calling the functions together in order to sort the coordinates
def SortingCoordinates(file):
    voxelData = read_vox_file(file)
    voxelSize = voxelData['size']
    voxels = voxelData['voxels']
    coords = np.delete(voxels, 3, 1)

    semiSortedCoordinates, xLayers = layeringRows(coords, 2)  # Layers each row based on Z Coordinate
    sortEachLayer(semiSortedCoordinates, xLayers)  # Sorts X Coordinates
    semiSortedCoordinates, yLayers = layeringRows(semiSortedCoordinates, 0)  # Layers each row based on X coordinate
    sortedCoordinates = sortEachLayer(semiSortedCoordinates, yLayers)  # Sorts Y Coordinates
    finalCoordinates = np.delete(sortedCoordinates, 4, 1)
    finalCoordinates = np.delete(finalCoordinates, 3, 1)

    return finalCoordinates, voxelSize


# inverts the lagrange numerically
def forward_robot(x, *jtvecin):
    r, s = x
    rs = r * s
    jtvec = jtvecin[0]
    n1 = .25 * rs * (1 - r) * (1 - s)
    n2 = -.25 * rs * (1 + r) * (1 - s)
    n3 = .25 * rs * (1 + r) * (1 + s)
    n4 = -.25 * rs * (1 - r) * (1 + s)
    n5 = -.5 * (1 - (r * r)) * s * (1 - s)
    n6 = .5 * (1 + r) * r * (1 - (s * s))  #functions to get multiplication factor for rs and xy values
    n7 = .5 * (1 - (r * r)) * s * (1 + s)
    n8 = -.5 * (1 - r) * r * (1 - (s * s))
    n9 = (1 - (r * r)) * (1 - (s * s))
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
              + jtvec[4][5] * n5 + jtvec[5][5] * n6 + jtvec[6][5] * n7 + jtvec[7][5] * n8 + jtvec[8][
                  5] * n9)  # Each joint angle calculation
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
    n1 = .25 * rs * (1 - r) * (1 - s)
    n2 = -.25 * rs * (1 + r) * (1 - s)
    n3 = .25 * rs * (1 + r) * (1 + s)
    n4 = -.25 * rs * (1 - r) * (1 + s)
    n5 = -.5 * (1 - (r * r)) * s * (1 - s)
    n6 = .5 * (1 + r) * r * (1 - (s * s))
    n7 = .5 * (1 - (r * r)) * s * (1 + s)
    n8 = -.5 * (1 - r) * r * (1 - (s * s))
    n9 = (1 - (r * r)) * (1 - (s * s))
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


fileName = input("Enter file name: ")
coordinates, size = SortingCoordinates(fileName)

"""
1 Starts at bottom left and goes around the corners, 
5 is in the middle of the bottom side and goes around the sides,
9 is center
Coordinates are in millimeters
"""
#Mapping Coordinates
ptvec = (
    [0, 0],
    [404, 0],
    [404, 404],
    [0, 404],
    [202, 0],
    [404, 202],
    [202, 404],
    [0, 202],
    [202, 202]
)
# Actual Coordinates
xyvec = (
    [0, 0],
    [404, 0],
    [404, 404],
    [0, 404],
    [202, 0],
    [404, 202],
    [202, 404],
    [0, 202],
    [202, 202]
)
# Joint Angles
jtvec = (
    [80.37, -57.26, -66.86, 0, -59.02, -9.12],
    [100.98, -57.56, -66.86, 0, -59.02, 11],
    [106.79, -35.48, -116.48, 0, -27.77, 16.68],
    [74.84, -35.48, -116.48, 0, -27.77, -9.12],
    [90.68, -56.23, -69.83, 0, -57.26, 1.64],
    [103.29, -45.02, -94.46, 0, -42.48, 12.88],
    [90.75, -35.31, -117.9, 0, -29.42, 1.64],
    [78.12, -45.02, -94.46, 0, -42.48, -5.40],
    [90.66, -44.49, -96.13, 0, -42.57, 2.71]
)

bcvec = []  # Pickup Block Coordinates
finaljtvec = []  # Final Joint Angles
finalxyvec = []  # Final XY Coordinates
voxelCoordinates = []
numberOfBlocks = len(coordinates)
intomm = 25.4

for point in coordinates:
    transformed_point = (
        float(point[0]) * intomm + intomm / 2 + 3,
        float(point[1]) * intomm + intomm / 2 + 3,
        float(point[2]) * intomm + intomm / 2
    )
    voxelCoordinates.append(transformed_point)

startx = intomm / 2
starty = 404 - intomm / 2
bcvec.append([startx, starty])

for i in range(1, numberOfBlocks, 1):
    if i <= 18:
        starty -= intomm * 1.35
        if i % 6 == 0:
            startx += intomm * 3
            starty = 404 - intomm / 2
    else:
        starty -= intomm * 1.35
        if i % 12 == 6:
            startx += intomm * 3
            starty = 404 - intomm / 2
    bcvec.append([startx, starty])

finalbcvec = []

for point in bcvec:
    x, j = map(point[0], point[1], ptvec, xyvec, jtvec)
    finalbcvec.append([round(num, 2) for num in j])

for point in voxelCoordinates:
    x, j = map(point[0], point[1], ptvec, xyvec, jtvec)
    finalxyvec.append([round(num, 2) for num in x])
    finaljtvec.append([round(num, 2) for num in j])


#server is opened which connects directly to the robot
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = "192.168.137.50"
port = 8085
server_socket.bind((host, port))

server_socket.listen(5)
client_socket, addr = server_socket.accept()


def received_message():
    recieved_data = client_socket.recv(1024)
    print(recieved_data)


closeGripper = b"close gripper"
openGripper = b"open gripper"
moveupvar = b"49"  #move up variable
movedownvar = b"-27"  #move down variable
moveup = [b"moveup", moveupvar]
movedown = [b"movedown", movedownvar]
rotatevar = b"90"
rotategripper = [b"rotate gripper", rotatevar]


def sendmessage(message):
    client_socket.send(b"start")
    time.sleep(0.7)
    if message[0] == b'moveto':
        client_socket.send(message[0])
        time.sleep(0.7)
        #received_message() # Small delay to ensure separation

        for jtangles in message[1]:
            time.sleep(0.7)
            client_socket.send(bytes(str(jtangles), 'utf-8'))
            time.sleep(0.7)
            #received_message() # Small delay to ensure separation
        client_socket.send(b'end')
        time.sleep(0.7)
    elif message[0] == b'moveup' or message[0] == b'movedown':
        client_socket.send(message[0])
        time.sleep(0.7)
        client_socket.send(message[1])
        time.sleep(0.7)
        client_socket.send(b'end')
    elif message[0] == b'rotate gripper':
        client_socket.send(message[0])
        time.sleep(.7)
        client_socket.send(message[1])
        time.sleep(.7)
        client_socket.send(b'end')
    else:
        client_socket.send(message)
        time.sleep(0.7)
        client_socket.send(b'end')
        time.sleep(0.7)
    return 0


x_coords = [coord[0] for coord in finalxyvec]
y_coords = [coord[1] for coord in finalxyvec]

sides = []

for i in range(numberOfBlocks):
    p1 = [finalxyvec[i][0], round(finalxyvec[i][1] - 25.4, 2)]
    p2 = [round(finalxyvec[i][0] - 25.4, 2), finalxyvec[i][1]]
    sides.append([p1,p2])





    

sendmessage(openGripper)
received_message()

for i in range(numberOfBlocks):

    moveToCube = [b'moveto', finalbcvec[i]]
    moveToPlace = [b'moveto', finaljtvec[i]]

    sendmessage(moveToCube)
    received_message()
    sendmessage(movedown)
    received_message()
    sendmessage(closeGripper)
    received_message()
    sendmessage(moveup)
    received_message()
    sendmessage(moveToPlace)
    received_message()
    print("Coords: ", finalxyvec[i])
    print("Sides: ", sides[i])
    if sides[i][0] in finalxyvec[:i] and sides[i][1] in finalxyvec[:i]:
        sendmessage(openGripper)
    elif sides[i][1] in finalxyvec[:i]:
        sendmessage(rotategripper)
        received_message()
        sendmessage(movedown)
        received_message()
        sendmessage(openGripper)
        received_message()
        sendmessage(rotategripper)
        received_message()
    else:
        sendmessage(movedown)
        received_message()
        sendmessage(openGripper)
        received_message()
