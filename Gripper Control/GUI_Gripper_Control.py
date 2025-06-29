
import tkinter as tk
from tkinter import messagebox
import serial
import threading
import time
import cv2
from ultralytics import YOLO

import serial.tools.list_ports
import matplotlib.pyplot as plt

import struct
from tkinter import ttk

model = YOLO("Gripper Control/nivea_grasp.pt").to('cuda')

closed_gripper = False
timer = time.time()



reading_ir = False
ir_value = None
ir_action_running = False
feedback_on = False


camera_running = False
camera_capture = None
camera_window_name = "USB Camera"
# Global variable to store the serial connection
arduino = None

# Port and baudrate can be changed if needed
PORT = 'COM5'
PORT = "/dev/ttyACM0" 
BAUDRATE = 9600

k =425



def find_arduino_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        # Look for common Arduino identifiers
        if "Arduino" in port.description or "ttyACM" in port.device or "ttyUSB" in port.device:
            return port.device
    return None



# === DEFAULT SERVO SETTINGS ===
default_servos = [
    {"name": "Finger Right", "id": "5", "angle": 78},
    {"name": "Finger Left", "id": "7", "angle": 96},
    {"name": "Thumb", "id": "8", "angle": 90},
    {"name": "Finger Right Phi", "id": "10", "angle":125},
    {"name": "Finger Left Phi", "id": "12", "angle": 85},
    {"name": "Thumb Phi", "id": "14", "angle": 80},

]

Gripper_State_Date = {
    "IR" : None,
    "pressure_left": None,
    "angle_left": None,
    "pressure_right": None,
    "angle_right": None,
    "pressure_thumb": None,
    "angle_thumb": None,
    
}

def init_servos():
    set_servo(5, 130-5)
    time.sleep(0.01)
    set_servo(7, 130)
    time.sleep(0.01)
    set_servo(9, 130)
    time.sleep(0.01)
    set_servo(10, 100)
    time.sleep(0.01)
    set_servo(12, 45)
    time.sleep(0.01)
    set_servo(14, 80)



def toggle_connection():
    global arduino
    PORT = find_arduino_port()
    if arduino is None or not arduino.is_open:
        try:
            arduino = serial.Serial(port=PORT, baudrate=BAUDRATE, timeout=0.1)
            status_label.config(text="Connected", fg="green")
            connect_button.config(text="Disconnect")
            time.sleep(0.1)
            init_servos()
        except serial.SerialException as e:
            messagebox.showerror("Connection Error", str(e))
    else:
        stop_data_reading()
        arduino.close()
        status_label.config(text="Disconnected", fg="red")
        connect_button.config(text="Connect")
        

def toggle_ir_action():
    global ir_action_running
    ir_action_running = not ir_action_running

    if ir_action_running:
        ir_action_button.config(text="Stop IR Action")
        run_ir_action_loop()
    else:
        ir_action_button.config(text="Start IR Action")
        
def run_ir_action_loop():
    global ir_action_running
    if not ir_action_running:
        return

    if ir_value is not None:
        print(f"IR Action using value: {ir_value}")
        if ir_value <=10:
            close_gripper()
            ir_action_running = False
            ir_action_button.config(text="Start IR Action")
            
        # Example: control a servo or trigger something
        # set_servo(5, map_ir_to_angle(ir_value))

    # Re-run after 100ms
    root.after(100, run_ir_action_loop)
        

   
def data_reader_loop():
    global reading_ir, ir_value, Gripper_State_Date, feedback_on 
    while reading_ir:
        read_data()
        if Gripper_State_Date["IR"]  is not None:
            
            voltage= Gripper_State_Date["IR"] * (5/1023)
            ir_value = 27.86 * (voltage**(-1.15))
            ir_label.config(text=f"IR Value: {ir_value}")
            print(Gripper_State_Date)
            # Example update

            
        #time.sleep(0.1)  # 10x per second
        
def start_data_reading():
    global reading_ir
    if not reading_ir:
        reading_ir = True
        threading.Thread(target=data_reader_loop, daemon=True).start()

def stop_data_reading():
    global reading_ir
    reading_ir = False
        
def read_data_string():
    if arduino and arduino.in_waiting > 0:
        line = arduino.readline().decode('utf-8').strip()
        if line.startswith("Data,"):
            try:
                parts = line.split(',')
              
                return int(parts[1]), int(parts[2]),int(parts[3]),int(parts[4]) ,int(parts[5]),int(parts[6]),int(parts[7])
            except (IndexError, ValueError):
                pass
    return None, None, None, None

#read data for binary mode
def read_data():
    
    global Gripper_State_Date
    
    PACKET_SIZE = 14  # 7 * 2-byte integers

    if arduino and arduino.in_waiting >= PACKET_SIZE:
        data = arduino.read(PACKET_SIZE)
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


        
        
#close gripper
def close_gripper():
    global feedback_on
    feedback_on = True

    angle = 83
    set_servo(7, angle)
 
    time.sleep(0.01)

    set_servo(5, angle-5)
    time.sleep(0.01)
    set_servo(8,angle)
    
#open gripper
def open_gripper():
    global closed_gripper, feedback_on
    feedback_on = False
    angle= 130
    set_servo(7, angle)

    time.sleep(0.01)
    set_servo(5, angle-5)
    time.sleep(0.01)
    set_servo(8, angle)

    closed_gripper = False
 
    
        
        
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
        
        messagebox.showwarning("Not Connected", "Connect to Arduino first.")
        
# === GUI CALLBACK ===
def basic_slider_change(entry_widget, angle):
    servo_id = entry_widget.get()
    if servo_id.strip().isdigit():
        set_servo(servo_id, int(float(angle)))
        entry_widget.config(bg="white")
    else:
        entry_widget.config(bg="misty rose")
        

def on_slider_change(changed_index, new_val):
    new_val = int(float(new_val))
    control = servo_controls[changed_index]

    # If NO checkboxes are selected → use normal mode
    if not any(c["sync_var"].get() for c in servo_controls):
        basic_slider_change(control["id_entry"], new_val)
        control["prev_val"] = new_val
        return

    # If THIS slider is not checked → ignore the move
    if not control["sync_var"].get():
        return

       # Set value for all checked sliders (including the one being moved)
    for i, c in enumerate(servo_controls):
        if c["sync_var"].get():
            c["slider"].set(new_val)
            c["prev_val"] = new_val
            servo_id = c["id_entry"].get().strip()
            if servo_id.isdigit():
                set_servo(servo_id, new_val)

        
def create_servo_control(row):
    # Frame for each servo control (entry + slider)
    frame = tk.Frame(control_frame)
    frame.grid(row=row, column=0, padx=10, pady=5, sticky="w")

    # Entry for servo ID
    id_entry = tk.Entry(frame, width=5)
    id_entry.grid(row=0, column=0, padx=(0, 10))

    # Slider for angle
    slider = tk.Scale(frame, from_=0, to=180, orient=tk.HORIZONTAL, length=200,
                      command=lambda val, entry=id_entry: on_slider_change(entry, val))
    slider.grid(row=0, column=1)

    return id_entry, slider 



def create_servo_row(row, name, servo_id, angle):
    frame = tk.Frame(control_frame)
    frame.grid(row=row, column=0, padx=10, pady=5, sticky="w")

    # Name Entry
    name_label = tk.Label(frame, text="Name:")
    name_label.grid(row=0, column=0)
    name_entry = tk.Entry(frame, width=10)
    name_entry.insert(0, name)
    name_entry.grid(row=0, column=1, padx=(0, 10))

    # Servo ID Entry
    id_label = tk.Label(frame, text="ID:")
    id_label.grid(row=0, column=2)
    id_entry = tk.Entry(frame, width=5)
    id_entry.insert(0, servo_id)
    id_entry.grid(row=0, column=3, padx=(0, 10))

    # Checkbox
    sync_var = tk.BooleanVar(value=False)
    checkbox = tk.Checkbutton(frame, variable=sync_var)
    checkbox.grid(row=0, column=4, padx=(0, 10))

    # Slider (no callback yet)
    slider = tk.Scale(frame, from_=0, to=180, orient=tk.HORIZONTAL, length=200)
    slider.set(angle)
    slider.grid(row=0, column=5)

    # Add callback after slider is set
    def slider_callback(val, index=len(servo_controls)):
        on_slider_change(index, val)

    slider.config(command=slider_callback)

    return {
        "name_entry": name_entry,
        "id_entry": id_entry,
        "slider": slider,
        "sync_var": sync_var,
        "prev_val": angle
    }


def toggle_camera():
    global camera_running, camera_capture

    if not camera_running:
        camera_capture = cv2.VideoCapture(0)
        if not camera_capture.isOpened():
            messagebox.showerror("Camera Error", "Unable to open camera.")
            return

        camera_running = True
        camera_button.config(text="Stop Camera")
        show_camera_frame()
    else:
        camera_running = False
        camera_button.config(text="Start Camera")
        if camera_capture:
            camera_capture.release()
        cv2.destroyWindow(camera_window_name)

def show_camera_frame():
    global camera_running, k, closed_gripper, timer

    if camera_running and camera_capture.isOpened():
        ret, frame = camera_capture.read()
        if ret:
            results = model(frame, verbose = False)
            results = model(frame, verbose = False)

    # Annotate the frame with the results
            annotated_frame = results[0].plot()
            frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            cv2.imshow(camera_window_name, annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            #= ord('q')
            pred_idx = results[0].probs.top1
            
            if pred_idx == 1 and closed_gripper == False and time.time()- timer >= 5:
                close_gripper()
                closed_gripper =True
                

          

            print("Predicted class:", pred_idx, closed_gripper)              


            if key== ord('s'):
                     #cv2.imwrite(f'yolo_training/training_data/grasp_position/negative/negative_{k}.jpg',frame)
                     cv2.imwrite(f'yolo_training/training_data/grasp_position/positive/positive_{k}.jpg',frame)
                     cv2.imwrite('test.jpg', frame)
                     print(f'img_{k}.jpg')
                     k+=1
                

        # Show next frame after 10 ms
        if cv2.getWindowProperty(camera_window_name, cv2.WND_PROP_VISIBLE) >= 1:
            root.after(10, show_camera_frame)
        else:
            # User manually closed window
            toggle_camera()
            
            
def update_table():
    # Clear current rows
    global Gripper_State_Date
    for row in tree.get_children():
        tree.delete(row)
    # Insert new rows
    for key, value in Gripper_State_Date.items():
        tree.insert("", "end", values=(key, value))
        
    root.after(100, update_table)


def on_closing():
    global arduino, reading_ir
    reading_ir = False
    if arduino and arduino.is_open:
        arduino.close()
        print("Serial connection closed.")
    root.destroy()
    
    

     

# Create the main window
root = tk.Tk()
root.title("Arduino Connector")
root.geometry("700x900")

root.protocol("WM_DELETE_WINDOW", on_closing)



# Status label
status_label = tk.Label(root, text="Disconnected", fg="red", font=("Arial", 12))
status_label.pack(pady=10)

# Connect/Disconnect button
connect_button = tk.Button(root, text="Connect", width=15, command=toggle_connection)
connect_button.pack(pady=10)

# Frame for all servo controls
control_frame = tk.Frame(root)
control_frame.pack(pady=10)


servo_controls = []
# Create 6 rows of servo controls
for i, servo in enumerate(default_servos):
    control = create_servo_row(i, servo["name"], servo["id"], servo["angle"])
    servo_controls.append(control)
    

ir_label = tk.Label(root, text="IR Value: --", font=("Arial", 14))
ir_label.pack(pady=10)

button_frame = tk.Frame(root)
button_frame.pack(pady=5)

gripper_frame = tk.Frame(root)
gripper_frame.pack(pady=10)

open_button = tk.Button(gripper_frame, text="Open Gripper", width=15, command=open_gripper)
open_button.grid(row=0, column=0, padx=5)

close_button = tk.Button(gripper_frame, text="Close Gripper", width=15, command=close_gripper)
close_button.grid(row=0, column=1, padx=5)

start_button = tk.Button(button_frame, text="Start Sensor Reading", command=start_data_reading)
start_button.grid(row=0, column=0, padx=5)

stop_button = tk.Button(button_frame, text="Stop Sensor Reading", command=stop_data_reading)
stop_button.grid(row=0, column=1, padx=5)

ir_action_button = tk.Button(root, text="Start IR Action", command=toggle_ir_action)
ir_action_button.pack(pady=5)

camera_button = tk.Button(root, text="Start Camera", command=toggle_camera)
camera_button.pack(pady=6)



##Add Data Table
columns = ("Parameter", "Value")
tree = ttk.Treeview(root, columns=columns, show="headings", height=7)
tree.heading("Parameter", text="Parameter")
tree.heading("Value", text="Value")
tree.pack(padx=10, pady=10)
update_table() 



# Start the UI loop
root.mainloop()
