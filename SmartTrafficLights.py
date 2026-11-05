# SmartTrafficLights.py
# Controller (Pi 1): runs YOLOv5n on lanes N and W, gets S and E data from Pi 2 over TCP,
# and drives the four traffic lights through GPIO.

import torch
import cv2
import time
from command_sender import CommandSender
import RPi.GPIO as GPIO
import re
import subprocess
import threading
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

Pi2 = CommandSender(target_ip = "PI2_IP_ADDRESS") #To be changed to pi 2 IP address
Pi2.connect() #connect to the local server made on raspberry pi 2

# Load YOLOv5n model for car counting
count_model = torch.hub.load('ultralytics/yolov5', 'custom', path='YOLOv5n_VehicleCount3.pt', force_reload=False)
count_model.conf = 0.75

#load YOLOv5n model for emergency vehicle detection
emergency_model = torch.hub.load('ultralytics/yolov5', 'custom', path='YOLOv5n_EmergencyVehicle3.pt', force_reload=False)
emergency_model.conf = 0.4

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
    "N": "usb-0:1.1:1.0",
    "W": "usb-0:1.2:1.0",
}

#class containing the useful variable concerning the lanes
class Lane:
    def __init__(self, cap, vehicle_count = 0, isEmergency = False, time_allocated = -1, results_count = None, results_emergency = None, start_time = -1):
        self.image = None
        self.vehicle_count = vehicle_count
        self.isEmergency = isEmergency
        self.time_allocated = time_allocated
        self.results_count = results_count
        self.results_emergency = results_emergency
        self.start_time = start_time
        self.cap = cap

    def isEmpty(self) -> bool:
        return self.vehicle_count == 0

#contains the functions needed to control the GPIO pins to change the lights
class TrafficControl:
    #GPIO mappings
    lights = {
    'N': {'G': 2, 'Y': 3, 'R': 4},
    'W': {'G': 14, 'Y': 15, 'R': 18},
    'S': {'G': 23, 'Y': 24, 'R': 25},
    'E': {'G': 17, 'Y': 27, 'R': 22}
    }

    @staticmethod
    def setup() -> None:
        GPIO.setmode(GPIO.BCM)
        for direction in TrafficControl.lights.values():    #interates through the dictionaries
            for pin in direction.values():                  #interates through the pins of said dictionary
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)

    @staticmethod
    def _set_all_red_except(active_key: str) -> None:
        for key, direction in TrafficControl.lights.items():
            if key != active_key:
                # Only turn off yellow and green
                GPIO.output(direction['Y'], GPIO.LOW)
                GPIO.output(direction['G'], GPIO.LOW)
                # Leave red on (or turn it on if not already)
                GPIO.output(direction['R'], GPIO.HIGH)

    @staticmethod
    def _set_color(dir_key: str, color: str) -> None:
        pins = TrafficControl.lights[dir_key]

        # Turn off only the other two lights
        for c in ['R', 'Y', 'G']:
            if c != color:
                GPIO.output(pins[c], GPIO.LOW)

        # Turn on the selected color
        GPIO.output(pins[color], GPIO.HIGH)

    @staticmethod
    def turnGreen(dir_key: str) -> None:
        if dir_key in TrafficControl.lights:
            TrafficControl._set_all_red_except(dir_key)
            TrafficControl._set_color(dir_key, 'G')

    @staticmethod
    def turnYellow(dir_key: str) -> None:
        if dir_key in TrafficControl.lights:
            TrafficControl._set_all_red_except(dir_key)
            TrafficControl._set_color(dir_key, 'Y')


#The lane where the emergency vehicle is turns green
#Then we constantly check for the emergency vehicle
def emergency(dirct: str) -> None:
    TrafficControl.turnGreen(dirct)
    while(lane[dirct].isEmergency):
        if dirct == 'N':
            take_image('N')
            check_emergency('N')
        elif dirct == 'W':
            take_image('W')
            check_emergency('W')
        elif dirct == 'S':
            lane['S'].isEmergency = Pi2.send('detect_emergencyS', expect_response=True)
        elif dirct == 'E':
            lane['E'].isEmergency = Pi2.send('detect_emergencyE', expect_response=True)
        time.sleep(1)

#takes images of the current traffic lanes
def take_image(dirct: str) -> bool:
    cap = cv2.VideoCapture(get_camera_index_by_usb_label(dirct))
    time.sleep(0.8)
    ret, frame = cap.read()
    cap.release()
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

def allocate_time(vehicle_count: int) -> int:
    if(all_lanes_empty()):
        return 5	#should be 20
    elif (vehicle_count < 9):
        return 10 	#should be 20
    elif (vehicle_count < 20):
        return 15 	#should be 35
    else:
        return 25 	#should be 50

def all_other_lanes_empty() -> bool:
    return all(lane[direction].isEmpty() for direction in lane if direction != current)

#returns True if all lanes are empty
def all_lanes_empty() -> bool:
    return all(lane[direction].isEmpty() for direction in lane)

def choose_next_lane() -> str:
    if all_other_lanes_empty():
        return current
    next_dir = next_lane[current]
    if all_lanes_empty():
        return next_dir
    while(lane[next_dir].isEmpty()):
        next_dir = next_lane[next_dir]
    return next_dir

def start_green_phase(dir_key):
    global current, green_timer, is_yellow_phase
    current = dir_key
    is_yellow_phase = False
    TrafficControl.turnGreen(current)
    lane[current].time_allocated = allocate_time(lane[current].vehicle_count)
    lane[current].start_time = time.time()
    print(f"[INFO] GREEN: {dir_key} for {lane[current].time_allocated}s")

    # Schedule transition to yellow
    green_timer = threading.Timer(lane[current].time_allocated, start_yellow_phase)
    green_timer.start()

def start_yellow_phase():
    global yellow_timer, green_timer, is_yellow_phase
    next_dir = choose_next_lane()

    # Executes when all other lanes than current are empty
    if current == next_dir:
        extended_time = 10
        print(f"[INFO] GREEN: {current} extended by 10s due to empty other lanes")
        green_timer = threading.Timer(extended_time, start_yellow_phase)
        green_timer.start()
        return

    is_yellow_phase = True
    # prepare to transition to another lane
    TrafficControl.turnYellow(current)
    print(f"[INFO] YELLOW: {current}, will switch soon")

    # Schedule next green
    yellow_timer = threading.Timer(yellow_light_delay, start_green_phase, args=[next_dir])
    yellow_timer.start()

def cancel_all_timers():
    global green_timer, yellow_timer
    if green_timer:
        green_timer.cancel()
        green_timer = None
    if yellow_timer:
        yellow_timer.cancel()
        yellow_timer = None


def print_lane_status():
    print("\n[STATUS] Lane Info:")
    print(f"{'Lane':<5} {'# Vehicles':<11} {'Emergency':<10}")
    print("-" * 30)
    for key in lane:
        vehicle_count = lane[key].vehicle_count
        is_em = "Yes" if lane[key].isEmergency else "No"
        print(f"{key:<5} {vehicle_count:<11} {is_em:<10}")


#each of the below entries signifies a different lane
#N, W, S, E signifies north lane, west lane, south lane, east lane
lane = {
    'N' : Lane(time_allocated=20, cap = None),
    'W' : Lane(cap = None),
    'E' : Lane(cap = None),
    'S' : Lane(cap = None)
}

annotated_frame = {'N': {'count':None, 'emergency':None}, 'W': {'count':None, 'emergency':None}}

#to know what lane is green
current = 'N'

#The traffic light turns counter-clockwise
next_lane = {
    'N': 'W',
    'W': 'S',
    'S': 'E',
    'E': 'N'
}

last_sample_time = -1
current_time = None
sample_time = 4

yellow_light_delay = 4

# Track the timers so we can cancel them
green_timer = None
yellow_timer = None
is_yellow_phase = False

# Start the initial green phase
TrafficControl.setup()
start_green_phase(current)

try:
    while True:

        # Requests the raspberry pi to begin inference
        Pi2.send('process_traffic', expect_response=False)

        last_sample_time = time.time()

        # Runs inference locally
        process_lane('N')
        time.sleep(0.2)
        process_lane('W')

        current_time = time.time()

        #give time for the other pi to finish processing
        if current_time - last_sample_time < sample_time:
            time.sleep(sample_time - (current_time - last_sample_time))

        #request the processed traffic data from the other pi
        lane['S'].vehicle_count, lane['S'].isEmergency, lane['E'].vehicle_count, lane['E'].isEmergency = Pi2.send('traffic_data', expect_response=True)

        print_lane_status()

        #check for any emergency vehicle in any lane
        for direction in lane:
            if lane[direction].isEmergency:
                cancel_all_timers()
                if direction != current:
                    TrafficControl.turnYellow(current)
                    time.sleep(yellow_light_delay)
                emergency(direction)
                if direction != current:
                    TrafficControl.turnYellow(direction)
                    time.sleep(yellow_light_delay)
                start_green_phase(choose_next_lane())



        current_time = time.time()

        # Early switch if lane is empty
        if (lane[current].vehicle_count == 0 and not all_lanes_empty()) and not is_yellow_phase:
            print(f"[INFO] Lane {current} is empty. Switching early.")
            cancel_all_timers()
            start_yellow_phase()

        cv2.waitKey(1)

except KeyboardInterrupt:
    print("\n[INFO] Stopped by user with KeyboardInterrupt.")

finally:
    #closing everything
    cv2.destroyAllWindows()
    Pi2.close()
    GPIO.cleanup()
    if green_timer != None:
        green_timer.cancel()
    if yellow_timer != None:
        yellow_timer.cancel()
