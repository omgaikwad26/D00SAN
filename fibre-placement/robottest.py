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
    [80.37, -57.56, -66.86, 0, -59.02, -9.12],
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
        float(point[0]) * intomm + intomm/2,
        float(point[1]) * intomm + intomm/2,
        float(point[2]) * intomm + intomm/2
    )
    voxelCoordinates.append(transformed_point)

startx = intomm/2
starty = 404 - intomm/2
bcvec.append([startx, starty])

"""
for i in range(numberOfBlocks):
    if i % 6 == 0:
        starty -= intomm * 2.5
        startx = intomm/2
    else:
        startx += intomm * 2.5
    bcvec.append([startx, starty])
"""

for i in range(1, numberOfBlocks, 1):
    if i % 6 == 0:
        starty = 404 - intomm/2
        startx -= intomm * 2.5
    else:
        startx -= 1.1 * intomm
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
    ###sendmessage(rotateGripper)
    ###recieve_message()
    sendmessage(movedown)
    received_message()
    sendmessage(openGripper)
    received_message()