from ultralytics import YOLO
import cv2
import numpy as np
import time

# Mouse callback function to capture 4 points and draw red dots
points = []
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append([x, y])
        print(f"Point {len(points)}: [{x}, {y}]")
        cv2.circle(param['frame'], (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Select 4 Points', param['frame'])
        if len(points) == 4:
            print("\nCollected 4 src_points for homography:")
            for i, pt in enumerate(points, 1):
                print(f"Point {i}: {pt[0]}.{pt[1]}")
            print("\nPlease provide corresponding dst_points (real-world coordinates in meters). Example:")
            example_dst = [[0, 0], [5, 0], [0, 10], [5, 10]]
            for i, pt in enumerate(example_dst, 1):
                print(f"Point {i}: {pt[0]}.{pt[1]}")

# Load YOLOv10 model
model = YOLO('yolov10n.pt')

# Function to open camera with specified index and backend
def open_camera(index, backend=cv2.CAP_DSHOW):
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        print(f"No camera at index {index}")
        return None
    print(f"Camera opened successfully at index {index}")
    return cap

# Try opening camera at different indices
camera_indices = [0, 1, 2,3,4,5,6,7]
cap = None
for index in camera_indices:
    cap = open_camera(index)
    if cap is not None:
        # Set high resolution and realistic frame rate
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)  # Set to 30 FPS for stability
        break

if cap is None:
    print("Error: Could not open any camera. Please check connections and drivers.")
    exit()

# Verify actual resolution and FPS
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Camera resolution: {frame_width}x{frame_height} at {fps} FPS")

# Verify frame reading
ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame from camera")
    cap.release()
    exit()
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start of video

# Define codec and create VideoWriter with high-quality settings
output_path = 'output_video_with_speed.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), isColor=True)

# Set up display window (non-fullscreen to avoid scaling issues)
cv2.namedWindow('YOLOv10 Vehicle Detection with Speed', cv2.WINDOW_NORMAL)
cv2.resizeWindow('YOLOv10 Vehicle Detection with Speed', frame_width, frame_height)

# Capture 4 points for homography
print("Press 'p' to pause and select 4 points...")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame during homography setup")
        break
    cv2.imshow('Press "p" to pause', frame)
    if cv2.waitKey(1) & 0xFF == ord('p'):
        break
frame_copy = frame.copy() if ret else None
if not ret or frame_copy is None:
    print("Error: No valid frame for homography point selection")
    cap.release()
    exit()

cv2.namedWindow('Select 4 Points', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Select 4 Points', frame_width, frame_height)
cv2.setMouseCallback('Select 4 Points', mouse_callback, {'frame': frame_copy})
print("Click 4 points on the video frame (e.g., road corners or lane markings). Press 'q' after selecting.")
while len(points) < 4:
    cv2.imshow('Select 4 Points', frame_copy)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyWindow('Select 4 Points')
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Homography setup
src_points = np.float32(points if len(points) == 4 else [
    [200, 600], [600, 600], [100, 100], [700, 100]
])

# Real-world coordinates in meters (adjust based on actual measurements)
dst_points = np.float32([
    [0, 0],
    [13, 0],  # bottom
    [0, 35],  # top
    [13, 35]
])

# Calculate homography matrix
H, _ = cv2.findHomography(src_points, dst_points)
if H is None:
    print("Error: Failed to compute homography matrix")
    cap.release()
    exit()

# Vehicle tracking information
vehicle_tracker = {}
next_vehicle_id = 0

# Number of frames for moving average
SPEED_HISTORY_SIZE = 30

# Minimum time difference to avoid division by near-zero
MIN_TIME_DIFF = 0.01

# Maximum realistic speed (km/h) to filter outliers
MAX_SPEED_KMPH = 200.0

# Function to transform pixel coordinates to real-world coordinates
def transform_point(point, H):
    point = np.array([point[0], point[1], 1], dtype=np.float32)
    transformed = np.dot(H, point)
    return transformed[:2] / transformed[2]

# Function to calculate distance in real-world coordinates
def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Function to calculate moving average speed
def calculate_moving_average(speed_history):
    if not speed_history:
        return None
    return sum(speed_history) / len(speed_history)

print("Press 's' to start processing...")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame during startup")
        break
    cv2.imshow('Press "s" to start', frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame")
            break

        start_time = time.time()
        current_frame_time = time.time()

        # Resize frame for YOLO processing to reduce computation
        yolo_frame = cv2.resize(frame, (416, 416))  # Smaller size for faster processing
        scale_x = frame_width / 416
        scale_y = frame_height / 416

        # Perform YOLO inference on resized frame
        results = model(yolo_frame, conf=0.45, iou=0.45)  # Adjust thresholds for performance

        current_detections = {}

        # Process detections
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Scale coordinates back to original frame size
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                current_center = ((x1 + x2) // 2, (y1 + y2) // 2)

                # Filter for vehicle classes
                if cls in [2, 3, 5, 7, 8]:  # car, motorcycle, bus, train, truck
                    label = f'{model.names[cls]} {conf:.2f}'

                    # Transform to real-world coordinates
                    current_center_world = transform_point(current_center, H)

                    # Check if vehicle is inside homography region
                    is_inside = cv2.pointPolygonTest(src_points, current_center, False) >= 0

                    # Find closest existing vehicle for tracking
                    matched_id = -1
                    min_distance = float('inf')
                    for vehicle_id, data in vehicle_tracker.items():
                        pixel_center = data['last_pixel_position']
                        dist = calculate_distance(current_center, pixel_center)
                        if dist < min_distance and dist < 100:
                            min_distance = dist
                            matched_id = vehicle_id

                    if matched_id != -1 and is_inside:
                        # Update existing vehicle
                        vehicle_data = vehicle_tracker[matched_id]
                        time_diff = current_frame_time - vehicle_data['last_frame_time']

                        if time_diff > MIN_TIME_DIFF:
                            last_position_world = vehicle_data['last_position_world']
                            meter_distance = calculate_distance(current_center_world, last_position_world)
                            speed_mps = meter_distance / time_diff
                            speed_kmph = speed_mps * 3.6

                            # Filter unrealistic speeds
                            if speed_kmph <= MAX_SPEED_KMPH:
                                vehicle_data['speed_history'].append(speed_kmph)
                                vehicle_data['speed_history'] = vehicle_data['speed_history'][-SPEED_HISTORY_SIZE:]
                                vehicle_data['speed'] = calculate_moving_average(vehicle_data['speed_history'])

                        vehicle_data['last_position_world'] = current_center_world
                        vehicle_data['last_pixel_position'] = current_center
                        vehicle_data['last_frame_time'] = current_frame_time
                        current_detections[matched_id] = vehicle_data

                    elif is_inside:
                        # New vehicle detected
                        vehicle_tracker[next_vehicle_id] = {
                            'last_position_world': current_center_world,
                            'last_pixel_position': current_center,
                            'last_frame_time': current_frame_time,
                            'speed': None,
                            'speed_history': []
                        }
                        current_detections[next_vehicle_id] = vehicle_tracker[next_vehicle_id]
                        next_vehicle_id += 1

                    # Draw bounding box
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Display speed if available and vehicle is inside homography region
                    if is_inside and matched_id != -1 and vehicle_tracker[matched_id]['speed'] is not None:
                        speed_text = f"ID: {matched_id} | Speed: {vehicle_tracker[matched_id]['speed']:.2f} km/h"
                        cv2.putText(frame, speed_text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        print(f"Vehicle ID: {matched_id}, Class: {model.names[cls]}, Confidence: {conf:.2f}, Speed: {vehicle_tracker[matched_id]['speed']:.2f} km/h")

        # Remove vehicles no longer detected
        vehicles_to_remove = [id for id in vehicle_tracker if id not in current_detections]
        for id in vehicles_to_remove:
            del vehicle_tracker[id]

        # Calculate and display FPS
        end_time = time.time()
        fps_actual = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0
        # cv2.putText(frame, f"FPS: {fps_actual:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write frame to output video
        out.write(frame)

        # Display frame
        cv2.imshow('YOLOv10 Vehicle Detection with Speed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing complete. Output video saved to {output_path}")