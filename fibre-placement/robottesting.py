
import cv2
import cvzone
import math
import time
from scipy.optimize import fsolve
import socket
import numpy as np
import matplotlib.pyplot as plt
"""
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
"""
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

for point in ptvec:
    coords = map(point[0],point[1], ptvec, xyvec, jtvec)
    print(coords[0])
    #x_min, y_min = coords[0][0]
    #x_max, y_max = coords[4][0]

    #print (x_min, y_min)
    #print (x_max, y_max)
    jtanglesStart = coords[1]
    print (jtanglesStart)

    




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


Robot goes to the first start position
moves down
draws the first line
moves up
Repeat 1-4


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