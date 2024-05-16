import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2
import cvzone
import math
import time
from scipy.optimize import fsolve
import socket

def get_mouse_click(event, x, y, flags, param):
    global click_count
    global click_coordinates
    if event == cv.EVENT_LBUTTONDOWN:
        click_coordinates.append((x,y))
        click_count +=1
        print(f'Click {click_count}: (x,y) = ({x}, {y})')
        if click_count == 4:
            cv.setMouseCallback('Calibration Feed', lambda *args:None) #Disables mouse callback after 4 clicks

def capture_calibration_points():
    global click_count, click_coordinates

    cap = cv.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Convert frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Corner detection
        corners = cv.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
        corners = np.int0(corners)

        for corner in corners:
            x, y = corner.ravel()
            cv.circle(frame, (x, y), 3, 255, -1)

        cv.imshow('Calibration Feed', frame)

        # Capture the first frame and process it
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

    # Now ask user to click on the detected corners
    click_count = 0
    click_coordinates = []

    cap = cv.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    cv.namedWindow('Calibration Feed')
    cv.setMouseCallback('Calibration Feed', get_mouse_click)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
# corner detection starts 
        # Convert the image to HSV color space
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # Define the color boundaries for blue in HSV space
        boundaries = [
            #([17, 15, 100], [50, 56, 200]),
            ([80,50,50], [130,255,255]),
            #([25, 146, 190], [62, 174, 250]),
            #([103, 86, 65], [145, 133, 128])
        ]

        # Initialize output
        output = None

        # Loop over the boundaries to detect specified colors
        for (lower, upper) in boundaries:
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
            
            # Create a mask for the color range
            mask = cv.inRange(hsv, lower, upper)
            
            # Apply the mask to the original image
            output = cv.bitwise_and(frame, frame, mask=mask)

        # Convert the masked output image to grayscale
        gray_output = cv.cvtColor(output, cv.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        gray_blurred = cv.GaussianBlur(gray_output, (9, 9), 0)

        # Use goodFeaturesToTrack to detect corners
        corners = cv.goodFeaturesToTrack(gray_blurred, maxCorners=20, qualityLevel=0.01, minDistance=10, blockSize=7)

        # Ensure corners are detected
        if corners is not None:
            corners = np.int0(corners)
            for i in corners:
                x, y = i.ravel()
                cv.circle(frame, (x, y), 5, (0, 255, 0), -1)

        cv.imshow('Calibration Feed', frame )

        if cv.waitKey(1) & 0xFF == ord('q') or click_count == 4:
            break

    cap.release()
    cv.destroyAllWindows()

    return click_coordinates

# Run the function and print the captured coordinates
click_coordinates = capture_calibration_points()
print(click_coordinates)  # The points displayed represent the calibration points

def forward_robot(x, *jtvecin):
    p,q = x
    jtvec = jtvecin[0]
    n1 = .25*(1-p)*(1-q) #functions to get multiplication factor for rs and xy values
    n2 = .25*(1+p)*(1-q)
    n3 = .25*(1+p)*(1+q)
    n4 = .25*(1-p)*(1+q)
    jt1val = jtvec[0][0]*n1+jtvec[1][0]*n2+jtvec[2][0]*n3+jtvec[3][0]*n4
    jt2val = jtvec[0][1]*n1+jtvec[1][1]*n2+jtvec[2][1]*n3+jtvec[3][1]*n4
    jt3val = jtvec[0][2]*n1+jtvec[1][2]*n2+jtvec[2][2]*n3+jtvec[3][2]*n4
    jt4val = jtvec[0][3]*n1+jtvec[1][3]*n2+jtvec[2][3]*n3+jtvec[3][3]*n4
    jt5val = jtvec[0][4]*n1+jtvec[1][4]*n2+jtvec[2][4]*n3+jtvec[3][4]*n4
    jt6val = jtvec[0][5]*n1+jtvec[1][5]*n2+jtvec[2][5]*n3+jtvec[3][5]*n4
    return[jt1val, jt2val, jt3val, jt4val, jt5val, jt6val]

def forward(x,*ptvecin): 
    #forward calculation of isoparametric mapping
    p,q = x
    ptvec = ptvecin[0]
    n1 = .25*(1-p)*(1-q) #functions to get multiplication factor for rs and xy values
    n2 = .25*(1+p)*(1-q)
    n3 = .25*(1+p)*(1+q)
    n4 = .25*(1-p)*(1+q)
    val1 = ptvec[0][0]*n1+ptvec[1][0]*n2+ptvec[2][0]*n3+ptvec[3][0]*n4 #finds the forward calculation for val1 and val2
    val2 = ptvec[0][1]*n1+ptvec[1][1]*n2+ptvec[2][1]*n3+ptvec[3][1]*n4  
    return[val1, val2]

def reverse_error(x,*args):
    rs = args[0][0]
    ptvec = args[0][1]
    x1 = forward(x,ptvec)
    err = [(x1[0]-rs[0]) * (x1[0]-rs[0]) , (x1[1]-rs[1]) * (x1[1]-rs[1])]
    return err

def reverse(r, s, *ptvec):
     data = [[r,s], ptvec[0]]
     x0 = [0.25,0.25]
     p = fsolve(reverse_error, x0, data)
     return p

def map(r,s,rspts,xypts, jtpts):
    p = reverse(r,s,rspts)
    x = forward(p, xypts)
    #j = forward_robot(p, jtpts)
    return x, j

#click coordinates are ptvec
ptvec = [[a,b] for a, b in click_coordinates]
xyvec = [[0,0], [686,0], [686,838], [0,838]]  #positions in mm
#jtvec = [[76.56, -79.99, -18.77, 0, -79.16, 0], [105.18, -79.99,-18.77, 0, -79.16,0], [126.06,-31.76,-129.71,0,-19.06, 0], [57.18,-31.76,-129.71,0,-19.06, 0]]


cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

with open("filepath.txt", "r") as f:
            model = YOLO(f.read())

success, img = cap.read()
results = model(img, stream=True)
for r in results:
    boxes = r.boxes
    for box in boxes:
        #bounding box
        r5, s5, r6, s6 = box.xyxy[0]                            #box.xywh - xy, width height
        r5, s5, r6, s6 = int(r5), int(s5), int(r6), int(s6)
        cv.rectangle(img, (r5, s5), (r6, s6),(255,0,255), 1)   #creates bounding box
        r_center = (r5 + r6)/2
        s_center = (s5 + s6)/2                                  #finds the center location of the bounding box
        coords = map(r_center, s_center, ptvec, xyvec, jtvec)
        print(coords)
        print("X: ", coords[0][0], ", Y: ", coords[1][1])         #displays x and y coordinates of the cube
        conf = math.ceil((box.conf[0]*100))/100             #confidence
        cls = int(box.cls[0])                               #class name
        cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, r5), max(35, s5)), scale=1, thickness=1) #displays conf and cls

cv.imshow("Image", img)
cv.waitKey(1)

jtangles = coords[1]

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

host = "192.168.137.50"
port = 8085
server_socket.bind((host, port))

server_socket.listen(5)
#print(f"Server listening on {host}:{port}...")
client_socket, addr = server_socket.accept()
#print(f"connection from {addr} established")

placeholder = b"40"
movetocube = [b"moveto", jtangles]
opengripper = [b"opengripper"]
closegripper = [b"closegripper"]
moveup = [b"moveup", placeholder]
movedown = [b"movedown", placeholder]

def sendmessage(message):
    client_socket.send(b"start")
    time.sleep(0.5)  # Small delay to ensure separation
    
    if message == b'moveto':
        client_socket.send(message[0])
        time.sleep(0.5)  # Small delay to ensure separation
        for jtangles in message[1]:
            time.sleep(0.5)
            client_socket.send(bytes(str(jtangles), 'utf-8'))
            time.sleep(0.5)  # Small delay to ensure separation
        client_socket.send(b'end')
    elif message == b'opengripper':
         client_socket.send(message)
    elif message == b'closegripper':
         client_socket.send(message)
    return 0


sendmessage(movetocube)
sendmessage(closegripper)
sendmessage(moveup)