
import tkinter as tk
from tkinter import messagebox
import serial
import threading
import time

reading_ir = False
ir_value = None
ir_action_running = False

# Global variable to store the serial connection
arduino = None

# Port and baudrate can be changed if needed
PORT = 'COM5'
BAUDRATE = 9600

# === DEFAULT SERVO SETTINGS ===
default_servos = [
    {"name": "Finger Right", "id": "7", "angle": 78},
    {"name": "Finger Left", "id": "5", "angle": 96},
    {"name": "Thumb", "id": "11", "angle": 90},
    {"name": "Finger Right Phi", "id": "12", "angle":125},
    {"name": "Finger Left Phi", "id": "14", "angle": 85},
    {"name": "Thumb Phi", "id": "9", "angle": 109},
]




def toggle_connection():
    global arduino
    if arduino is None or not arduino.is_open:
        try:
            arduino = serial.Serial(port=PORT, baudrate=BAUDRATE, timeout=0.1)
            status_label.config(text="Connected", fg="green")
            connect_button.config(text="Disconnect")
        except serial.SerialException as e:
            messagebox.showerror("Connection Error", str(e))
    else:
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
        
        
def ir_reader_loop():
    global reading_ir, ir_value
    while reading_ir:
        value = read_ir_data()
        if value is not None:
            
            voltage= value * (5/1023)
            ir_value = 27.86 * (voltage**(-1.15))
            ir_label.config(text=f"IR Value: {ir_value}")
        #time.sleep(0.1)  # 10x per second
        
def start_ir_reading():
    global reading_ir
    if not reading_ir:
        reading_ir = True
        threading.Thread(target=ir_reader_loop, daemon=True).start()

def stop_ir_reading():
    global reading_ir
    reading_ir = False
        
def read_ir_data():
    if arduino and arduino.in_waiting > 0:
        line = arduino.readline().decode('utf-8').strip()
        if line.startswith("IR,"):
            try:
                parts = line.split(',')
                return int(parts[1])
            except (IndexError, ValueError):
                pass
    return None

#close gripper
def close_gripper():
    set_servo(7, 70)
    time.sleep(0.01)

    set_servo(5, 70)
    time.sleep(0.01)

    set_servo(11,70)
    
#open gripper
def open_gripper():
    set_servo(7, 150)
    time.sleep(0.01)

    set_servo(5, 150)
    time.sleep(0.01)

    set_servo(11, 150)
        
        
def set_servo(servo_id, angle):
    print(servo_id)
    print('angle before', angle)
    if int(servo_id) == 5:
        angle = abs(angle-180)
        print('angle after', angle)
        
    if arduino and arduino.is_open:
        command = f"S,{servo_id},{angle}\n"
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
root.geometry("600x600")

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

start_button = tk.Button(button_frame, text="Start IR Reading", command=start_ir_reading)
start_button.grid(row=0, column=0, padx=5)

stop_button = tk.Button(button_frame, text="Stop IR Reading", command=stop_ir_reading)
stop_button.grid(row=0, column=1, padx=5)

ir_action_button = tk.Button(root, text="Start IR Action", command=toggle_ir_action)
ir_action_button.pack(pady=5)

# Start the UI loop
root.mainloop()
