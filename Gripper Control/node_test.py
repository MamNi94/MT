import serial
import serial.tools.list_ports
import struct

import time

# Global variable to store the serial connection
arduino = None

# Port and baudrate can be changed if needed
PORT = 'COM5'
PORT = "/dev/ttyACM0" 
BAUDRATE = 9600

'''
Angle Table
Finger Right

'''



Gripper_State_Date = {
    "IR" : None,
    "pressure_left": None,
    "angle_left": None,
    "pressure_right": None,
    "angle_right": None,
    "pressure_thumb": None,
    "angle_thumb": None,
    
}


def set_servo(servo_id, angle, pressure = None):
    print(servo_id)
    print('angle before', angle)
    if int(servo_id) == 7:
   
        angle = abs(angle-180)
       
        
    if arduino and arduino.is_open:
        if pressure is None:
            command = f"S,{servo_id},{angle}\n"
        else:
            command = f"S,{servo_id},{angle},{pressure}\n"
        arduino.write(command.encode())
       
    else:
        
        print("Not Connected", "Connect to Arduino first.")


def init_servos():
    #Init Finger Right
    set_servo(5, 130-5) 
    time.sleep(0.01)
    #Init Finger Left
    set_servo(7, 130)
    time.sleep(0.01)
    #Init Thumb
    set_servo(8, 130)
    time.sleep(0.01)
    #Init Finger Right Phi
    set_servo(10, 100)
    time.sleep(0.01)
    #Init Finger Left Phi
    set_servo(12, 45)
    time.sleep(0.01)
    #Init Thumb Phi
    set_servo(14, 74)


def find_arduino_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        # Look for common Arduino identifiers
        if "Arduino" in port.description or "ttyACM" in port.device or "ttyUSB" in port.device:
            return port.device
    return None


#read data for binary mode
def read_data():
    
    global Gripper_State_Date
    print("reading data")
    print("in waiting", arduino.in_waiting)
    
    PACKET_SIZE = 14  # 7 * 2-byte integers

    if arduino and arduino.in_waiting >=PACKET_SIZE :
        data = arduino.read(PACKET_SIZE)
        print("data", data)
        if len(data) == PACKET_SIZE:
            try:
                # '<7h' = little-endian, 7 short integers
                values = struct.unpack('<7h', data)
                #print("Received:", values)
                # Return first 4 values, or modify as needed
                keys = list(Gripper_State_Date.keys())
                for key, value in zip(keys, values):
                    if key == "angle_left":
                        Gripper_State_Date[key] = abs(value-180)
                        
                    else:
                        Gripper_State_Date[key] = value
                
                
                return
            except struct.error:
                pass
    return 



PORT = find_arduino_port()
print("port", PORT)

if arduino is None or not arduino.is_open:
        try:
            arduino = serial.Serial(port=PORT, baudrate=BAUDRATE, timeout=0.1)
          
            time.sleep(0.1)
            print("arduino connected")
            init_servos()
            
        except serial.SerialException as e:
            print("Connection Error", str(e))


i = 0

while i <= 10:
    #print(arduino)
    time.sleep(0.1)
    if arduino:
        read_data()
        print(Gripper_State_Date)
        
        i+=1

arduino.close()