# command_listener.py
# Worker (Pi 2): runs YOLOv5n on lanes S and E and answers controller commands over TCP.

#code for receiving and executing commands
#sometimes returning a response
#doesnt send data alone
# execute on pi 2

import socket
import time
import torch
import cv2
import pickle
import subprocess
import re
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Load YOLOv5n model for car counting
count_model = torch.hub.load('ultralytics/yolov5', 'custom', path='YOLOv5n_VehicleCount3.pt', force_reload=False)
count_model.conf = 0.75

#load YOLOv5n model for emergency vehicle detection
emergency_model = torch.hub.load('ultralytics/yolov5', 'custom', path='YOLOv5n_EmergencyVehicle3.pt', force_reload=False)
emergency_model.conf = 0.3

def get_camera_index_by_usb_label(label: str) -> int:
    expected_usb = USB_CAMERAS.get(label)
    if expected_usb is None:
        raise ValueError(f"No USB mapping found for label: {label}")

    # Loop through /dev/video* devices
    for i in range(24):  #iterate through the 2 cameras
        dev_path = f"/dev/video{i}"
        try:
            info = subprocess.check_output(
                ["udevadm", "info", "--query=all", "--name", dev_path],
                text=True
            )
            match = re.search(r'ID_PATH=(.+)', info)
            if match and expected_usb in match.group(1):
                return i  # This index can be passed to cv2.VideoCapture(i)
        except subprocess.CalledProcessError:
            continue

    raise RuntimeError(f"Camera with USB path {expected_usb} not found.")

# Hardcoded expected USB port paths for each function
USB_CAMERAS = {
    "S": "usb-0:1.1:1.0",
    "E": "usb-0:1.2:1.0",
}

class Lane:
    def __init__( self, cap, vehicle_count = 0, isEmergency = False, results_count = None, results_emergency = None):
        self.image = None
        self.cap = cap
        self.vehicle_count = vehicle_count
        self.isEmergency = isEmergency
        self.results_count = results_count
        self.results_emergency = results_emergency

#takes images of the current traffic lanes
def take_image(dirct: str) -> bool:
    cap = cv2.VideoCapture(get_camera_index_by_usb_label(dirct))
    time.sleep(0.8)
    ret, frame = cap.read()
    cap.release
    if not ret:
        return False
    else:
        lane[dirct].image = frame
        return True

#process the images and updates the car count in the dedicated lane
def count_cars(dirct : str):
    start = time.time()
    lane[dirct].results_count = count_model(lane[dirct].image)
    end = time.time()

    print(f"Inference Time (Count) (Lane {dirct}): {(end - start)*1000:.2f} ms")

    # Draw results on frame
    annotated_frame[dirct]['count'] = lane[dirct].results_count.render()[0]

    lane[dirct].vehicle_count = len(lane[dirct].results_count.pred[0]) if lane[dirct].results_count.pred[0] is not None else 0

    # Show result
    cv2.imshow(("YOLOv5n Vehicle Detection  in lane" + " " + dirct), annotated_frame[dirct]['count'])
    cv2.waitKey(1)  # Add after each cv2.imshow()

def check_emergency(dirct : str):
    start = time.time()
    lane[dirct].results_emergency = emergency_model(lane[dirct].image)
    end = time.time()

    print(f"Inference Time (Emergency) (Lane {dirct}): {(end - start)*1000:.2f} ms")

    # Draw results on frame
    annotated_frame[dirct]['emergency'] = lane[dirct].results_emergency.render()[0]

    lane[dirct].isEmergency =  len(lane[dirct].results_emergency.pred[0]) > 0

    # Show result
    cv2.imshow(("YOLOv5n Emergency Detection in lane" + " " + dirct), annotated_frame[dirct]['emergency'])
    cv2.waitKey(1)  # Add after each cv2.imshow()

def process_lane(dirct: str) -> None:
    if not take_image(dirct):
        raise RuntimeError(f"[ERROR] Failed to capture image for lane {dirct}")
    count_cars(dirct)
    check_emergency(dirct)


lane = {'S': Lane(cap = None),
        'E': Lane(cap = None)}

annotated_frame = {'S': {'count':None, 'emergency':None}, 'E': {'count':None, 'emergency':None}}

def process_traffic():
    process_lane('E')
    time.sleep(0.2)
    process_lane('S')
    return None

def send_traffic_data():
    return (lane['S'].vehicle_count, lane['S'].isEmergency, lane['E'].vehicle_count, lane['E'].isEmergency)

def detect_emergencyS():
    if not take_image('S'):
        raise RuntimeError(f"Camera S not open or failed to capture image.")
    check_emergency('S')
    return lane['S'].isEmergency

def detect_emergencyE():
    if not take_image('E'):
        raise RuntimeError(f"Camera E not open or failed to capture image.")
    check_emergency('E')
    return lane['E'].isEmergency

command_map = {'process_traffic': process_traffic,
               'traffic_data': send_traffic_data,
               'detect_emergencyS': detect_emergencyS,
               'detect_emergencyE': detect_emergencyE}


# Configuration
HOST = ''           # Listen on all interfaces
PORT = 6001         # Pick a port for incoming command connection
BUFFER_SIZE = 1024

def start_listener():
    # Set up a TCP listener
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((HOST, PORT))
    sock.listen(1)

    print(f"[+] Command listener started on port {PORT}...")

    conn, addr = sock.accept()
    print(f"[+] Connected to command sender at {addr}")

    while True:
        try:
            data = conn.recv(BUFFER_SIZE)
            if not data:
                print("[-] Connection closed by sender")
                break

            command = data.decode()
            print(f"[Command Received] {command}")

            # Check if this command expects a response
            expects_response = command.startswith("REQUIRES_RESPONSE:")
            actual_command = command[len("REQUIRES_RESPONSE:"):].strip() if expects_response else command.strip()

            # Execute if the command is valid
            if actual_command in command_map:
                result = command_map[actual_command]()  # execute the function
                if expects_response:
                    response = pickle.dumps(result)
                    conn.sendall(response)
                    print(f"[Sent Response] {response}")
                else:
                    print(f"[Executed without reply] {actual_command}")
            else:
                error_msg = f"Error: Unknown command '{actual_command}'"
                if expects_response:
                    conn.sendall(error_msg.encode())
                print(f"[Error] {error_msg}")


        except Exception as e:
            print(f"[Error] {e}")
            break

    conn.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_listener()
