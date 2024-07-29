### READ BELOW
### below is the code that is uploaded into the Doosan robot to work with the server, once the server is running commands can be sent 
### which correspond to different functions of the robot
import time
sock = client_socket_open("192.168.137.50", 8085) 
#client_socket_state(sock) 

def receive_message():
    payloads = []
    try:
        res, rx_data = client_socket_read(sock)   
        start = rx_data
        #client_socket_write(sock, b"received start")
        res, rx_data = client_socket_read(sock)   
        command = rx_data
        #client_socket_write(sock, b"received command")
        res, rx_data = client_socket_read(sock)
        # Receive and process messages until "end" is received
        #client_socket_write(sock, b"received payload")
        while rx_data != b'end':
            try:  
                payloads.append(int(rx_data))
                #client_socket_write(sock, b"received payload 1")
            except ValueError:    
                payloads.append(rx_data)
                #client_socket_write(sock, b"received payload with value error")
            res, rx_data = client_socket_read(sock)
        return command, payloads
    except Exception as e:
        #tp_popup("An error occurred:", e)
        return None, []

def movetoStart(payloads):
    float_payloads = [float(payload.decode('utf-8')) for payload in payloads[:6]]
    point_a = posj(float_payloads[0], 
                   float_payloads[1], 
                   float_payloads[2], 
                   float_payloads[3], 
                   float_payloads[4], 
                   float_payloads[5])
    movej(point_a, v=15, a=30, r=0)
    client_socket_write(sock, b"moved to start point")

def movetoEnd(payloads):
    float_payloads = [float(payload.decode('utf-8')) for payload in payloads[6:12]]
    point_b = posx(float_payloads[0], 
                   float_payloads[1], 
                   float_payloads[2], 
                   float_payloads[3], 
                   float_payloads[4], 
                   float_payloads[5])
    movel(point_b, v=15, a=30)
    client_socket_write(sock, b"moved to end point")

while True:
    command, payloads = receive_message()
    if command is None:
        #tp_popup("Error receiving message. Exiting...")
        time.sleep(1)
    #    break
    if command == b"moveto":
        if len(payloads) < 12:
            client_socket_write(sock, b"invalid payload length")
            continue
        
        movetoStart(payloads)
        movetoEnd(payloads)
    elif command == b"open gripper":
        set_tool_digital_output(1, OFF)
        client_socket_write(sock, b"opened gripper")
    elif command == b"close gripper":
        set_tool_digital_output(1, ON)
        client_socket_write(sock, b"closed gripper")
    elif command == b"moveup":
        height = float(payloads[0].decode('utf-8'))
        payloads = posx(0,0,height,0,0,0)
        movel(payloads, v=15, a=30, mod=DR_MV_MOD_REL)
        client_socket_write(sock, b"moved up")
    elif command == b"movedown":
        height = float(payloads[0].decode('utf-8'))
        payloads = posx(0,0,-height,0,0,0)
        movel(payloads, v=15, a=30, mod=DR_MV_MOD_REL)
        client_socket_write(sock, b"moved down")
    elif command == b"rotate gripper":
        distance = float(payloads[0].decode('utf-8'))
        payloads = posx(0,0,0,0,0,distance)
        movel(payloads, v=15, a=30, mod=DR_MV_MOD_REL)
        client_socket_write(sock, b"rotated gripper")
    else:
        client_socket_write(sock, b"command not processed")
