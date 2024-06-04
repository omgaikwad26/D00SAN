from ultralytics import YOLO
import cv2
import cvzone
import math
import time
from scipy.optimize import fsolve
import socket

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
    jt6val = jtvec[0][5]*n1+jtvec[1][5]*n2+jtvec[2][5]*n3+jtvec[3][5]*n4 #Each joint angle calculation
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
    j = forward_robot(p, jtpts)
    return x, j

#click coordinates are ptvec
ptvec = [[a,b] for a, b in click_coordinates]
xyvec = [[0,0], [686,0], [686,838], [0,838]]  #positions in mm
jtvec = [[76.56, -79.99, -18.77, 0, -79.16, 0], [105.18, -79.99,-18.77, 0, -79.16,0], [126.06,-31.76,-129.71,0,-19.06, 0], [57.18,-31.76,-129.71,0,-19.06, 0]]
#Vector points, real world coordinates, and joint angles at each corner for isoparametric mapping


#AI Cube Detection and position mapping
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

with open("filepath.txt", "r") as f:
            model = YOLO(f.read())


classNames = ["Purple Cube", "Blue Cube", "Green Cube", "Red Cube"]

success, img = cap.read()
results = model(img, stream=True)
for r in results:
    boxes = r.boxes
    for box in boxes:
        #bounding box
        r5, s5, r6, s6 = box.xyxy[0]                            #box.xywh - xy, width height
        r5, s5, r6, s6 = int(r5), int(s5), int(r6), int(s6)
        cv2.rectangle(img, (r5, s5), (r6, s6),(255,0,255), 1)   #creates bounding box
        r_center = (r5 + r6)/2
        s_center = (s5 + s6)/2                                  #finds the center location of the bounding box
        coords = map(r_center, s_center, ptvec, xyvec, jtvec)
        print(coords)
        print("X: ", coords[0][0], ", Y: ", coords[1][1])         #displays x and y coordinates of the cube
        conf = math.ceil((box.conf[0]*100))/100             #confidence
        cls = int(box.cls[0])                               #class name
        cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, r5), max(35, s5)), scale=1, thickness=1) #displays conf and cls


jtangles = coords[1]
cv2.imshow("Image", img)
cv2.waitKey(1)

#server is opened which connects directly to the robot 
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = "192.168.137.50"
port = 8085
server_socket.bind((host, port))

server_socket.listen(5)
client_socket, addr = server_socket.accept()

jtangles_for_sort = [100.81, -59.9, -63.47, 0.0, -55.51, 0.0] #map(821, 463, ptvec, xyvec, jtvec)
moveupvar = b"40" #move up variable
movedownvar= b"35" #move down variable
movetocube = [b"moveto", jtangles]          #message sent to robot that has joint positions calculated through isoparametric mapping
movetosort = [b"moveto", jtangles_for_sort] #message sent to robot that contains the joint positions for the robot to "sort"
opengripper = b"open gripper"
closegripper = b"close gripper"
moveup = [b"moveup", moveupvar]
movedown = [b"movedown", movedownvar]


def received_message():
    recieved_data = client_socket.recv(1024)
    print(recieved_data)

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
    else:
        client_socket.send(message)
        time.sleep(0.7)
        client_socket.send(b'end')
        time.sleep(0.7)
    return 0

sendmessage(movetocube)
received_message()
sendmessage(closegripper)
received_message()
sendmessage(moveup)
received_message()
sendmessage(movetosort)
received_message()
sendmessage(opengripper)
