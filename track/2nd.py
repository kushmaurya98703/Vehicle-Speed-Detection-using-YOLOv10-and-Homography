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
        # Draw red dot on the frame
        cv2.circle(param['frame'], (x, y), 5, (0, 0, 255), -1)  # Red dot, radius 5, filled
        cv2.imshow('Select 4 Points', param['frame'])
        if len(points) == 4:
            print("\nCollected 4 src_points for homography:")
            for i, pt in enumerate(points, 1):
                print(f"Point {i}: {pt[0]}.{pt[1]}")
            print("\nPlease provide corresponding dst_points (real-world coordinates in meters). Example:")
            example_dst = [[0, 0], [5, 0], [0, 10], [5, 10]]
            for i, pt in enumerate(example_dst, 1):
                print(f"Point {i}: {pt[0]}.{pt[1]}")

# Load a pre-trained YOLOv10 model
model = YOLO('yolov10n.pt')  # You can choose yolov10s.pt, yolov10m.pt, yolov10b.pt, yolov10l.pt, yolov10x.pt for different sizes

# Open the video file
video_path = 'video-1.mp4'  # Replace with your video path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
output_path = 'output_video_with_speed.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Set up full-screen display window
cv2.namedWindow('YOLOv10 Vehicle Detection with Speed', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('YOLOv10 Vehicle Detection with Speed', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Capture 4 points for homography
ret, frame = cap.read()
if ret:
    frame_copy = frame.copy()  # Copy frame to draw dots on
    cv2.namedWindow('Select 4 Points', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Select 4 Points', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback('Select 4 Points', mouse_callback, {'frame': frame_copy})
    print("Click 4 points on the video frame (e.g., road corners or lane markings). Press 'q' after selecting.")
    while len(points) < 4:
        cv2.imshow('Select 4 Points', frame_copy)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow('Select 4 Points')
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to start

# Homography setup: Define 4 points in pixel and real-world coordinates
# Use captured points or default if not enough points collected
src_points = np.float32(points if len(points) == 4 else [
    [200, 600],  # Bottom-left
    [600, 600],  # Bottom-right
    [100, 100],  # Top-left
    [700, 100]   # Top-right
])

# Corresponding real-world coordinates in meters (replace with actual measurements)
dst_points = np.float32([
    [0, 0],      # Bottom-left
    [500, 0],      # Bottom-right (e.g., 500 meters wide)
    [0, 100],     # Top-left (e.g., 100 meters up)
    [500, 100]      # Top-right
])

# Calculate homography matrix
H, _ = cv2.findHomography(src_points, dst_points)

# Store vehicle tracking information
# {id: {'last_position': (x, y), 'last_position_world': (x, y), 'last_frame_time': time, 'speed': None, 'speed_history': [], 'pixels_per_meter': float}}
vehicle_tracker = {}
next_vehicle_id = 0

# Number of frames for moving average
SPEED_HISTORY_SIZE = 5

# Average vehicle width in meters for dynamic PIXELS_PER_METER calculation
AVERAGE_VEHICLE_WIDTH = 2.0  # Typical car width in meters

# Function to transform pixel coordinates to real-world coordinates
def transform_point(point, H):
    point = np.array([point[0], point[1], 1], dtype=np.float32)
    transformed = np.dot(H, point)
    return transformed[:2] / transformed[2]

# Function to calculate distance between two points in real-world coordinates
def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Function to calculate moving average speed
def calculate_moving_average(speed_history):
    if not speed_history:
        return None
    return sum(speed_history) / len(speed_history)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_frame_time = time.time()

    # Perform inference on the frame
    results = model(frame)

    current_detections = {}

    # Process results and draw bounding boxes
    for r in results:
        boxes = r.boxes  # Bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            current_center = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Filter for vehicle classes
            if cls in [2, 3, 5, 7, 8]:  # car, motorcycle, bus, train, truck
                label = f'{model.names[cls]} {conf:.2f}'

                # Transform current center to real-world coordinates
                current_center_world = transform_point(current_center, H)

                # Check if vehicle center is inside homography region
                is_inside = cv2.pointPolygonTest(src_points, current_center, False) >= 0

                # Simple tracking: find the closest existing vehicle
                matched_id = -1
                min_distance = float('inf')
                for vehicle_id, data in vehicle_tracker.items():
                    dist = calculate_distance(current_center, data['last_position'])
                    if dist < min_distance and dist < 50:  # Threshold for matching
                        min_distance = dist
                        matched_id = vehicle_id

                if matched_id != -1:
                    # Update existing vehicle
                    vehicle_data = vehicle_tracker[matched_id]

                    # Calculate speed in real-world coordinates (homography-based)
                    time_diff = current_frame_time - vehicle_data['last_frame_time']
                    if time_diff > 0:
                        last_position_world = vehicle_data['last_position_world']
                        meter_distance = calculate_distance(current_center_world, last_position_world)
                        speed_mps = meter_distance / time_diff  # Speed in meters per second
                        speed_kmph = speed_mps * 3.6  # Convert to km/h

                        # Calculate speed in pixel coordinates
                        pixel_distance = calculate_distance(current_center, vehicle_data['last_position'])
                        pixels_per_meter = vehicle_data.get('pixels_per_meter', 10.0)  # Use stored PPM or default
                        meter_distance_pixel = pixel_distance / pixels_per_meter
                        speed_mps_pixel = meter_distance_pixel / time_diff
                        speed_kmph_pixel = speed_mps_pixel * 3.6

                        # Merge homography and pixel-based speeds (weighted average)
                        speed_kmph_final = 0.5 * speed_kmph + 0.5 * speed_kmph_pixel

                        # Update speed history
                        if 'speed_history' not in vehicle_data:
                            vehicle_data['speed_history'] = []
                        vehicle_data['speed_history'].append(speed_kmph_final)
                        # Keep only the last SPEED_HISTORY_SIZE values
                        vehicle_data['speed_history'] = vehicle_data['speed_history'][-SPEED_HISTORY_SIZE:]
                        # Calculate smoothed speed
                        vehicle_data['speed'] = calculate_moving_average(vehicle_data['speed_history'])

                    vehicle_data['last_position'] = current_center
                    vehicle_data['last_position_world'] = current_center_world
                    vehicle_data['last_frame_time'] = current_frame_time
                    current_detections[matched_id] = vehicle_data

                else:
                    # New vehicle detected
                    # Calculate dynamic PIXELS_PER_METER based on bounding box width
                    pixel_width = abs(x2 - x1)
                    pixels_per_meter = pixel_width / AVERAGE_VEHICLE_WIDTH if pixel_width > 0 else 10.0  # Fallback to default

                    vehicle_tracker[next_vehicle_id] = {
                        'last_position': current_center,
                        'last_position_world': current_center_world,
                        'last_frame_time': current_frame_time,
                        'speed': None,
                        'speed_history': [],
                        'pixels_per_meter': pixels_per_meter
                    }
                    current_detections[next_vehicle_id] = vehicle_tracker[next_vehicle_id]
                    next_vehicle_id += 1

                # Draw bounding box
                color = (0, 255, 0)  # Green
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw speed and print to console only if vehicle is inside homography region
                if is_inside and matched_id != -1 and vehicle_tracker[matched_id]['speed'] is not None:
                    speed_text = f"Speed: {vehicle_tracker[matched_id]['speed']:.2f} km/h"
                    cv2.putText(frame, speed_text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    # Print to console
                    print(f"Vehicle ID: {matched_id}, Class: {model.names[cls]}, Confidence: {conf:.2f}, Speed: {vehicle_tracker[matched_id]['speed']:.2f} km/h")

    # Remove vehicles that are no longer detected
    vehicles_to_remove = [id for id in vehicle_tracker if id not in current_detections]
    for id in vehicles_to_remove:
        del vehicle_tracker[id]

    # Write the frame to the output video
    out.write(frame)

    # Display the frame in full-screen
    cv2.imshow('YOLOv10 Vehicle Detection with Speed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Processing complete. Output video saved to {output_path}")