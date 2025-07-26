from ultralytics import YOLO
import cv2
import numpy as np
import time

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

# Store vehicle tracking information
# {id: {'last_position': (x, y), 'last_frame_time': time, 'speed': None}}
vehicle_tracker = {}
next_vehicle_id = 0

# Calibration: pixels per meter (PPM)
PIXELS_PER_METER = 10.0  # Example value, needs calibration

# Function to calculate distance between two points
def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

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

                    # Calculate speed
                    time_diff = current_frame_time - vehicle_data['last_frame_time']
                    if time_diff > 0:
                        pixel_distance = calculate_distance(current_center, vehicle_data['last_position'])
                        meter_distance = pixel_distance / PIXELS_PER_METER
                        speed_mps = meter_distance / time_diff  # Speed in meters per second
                        speed_kmph = speed_mps * 3.6  # Convert to km/h
                        vehicle_data['speed'] = speed_kmph

                    vehicle_data['last_position'] = current_center
                    vehicle_data['last_frame_time'] = current_frame_time
                    current_detections[matched_id] = vehicle_data

                else:
                    # New vehicle detected
                    vehicle_tracker[next_vehicle_id] = {
                        'last_position': current_center,
                        'last_frame_time': current_frame_time,
                        'speed': None
                    }
                    current_detections[next_vehicle_id] = vehicle_tracker[next_vehicle_id]
                    next_vehicle_id += 1

                # Draw bounding box and speed
                color = (0, 255, 0)  # Green
                if matched_id != -1 and vehicle_tracker[matched_id]['speed'] is not None:
                    speed_text = f"Speed: {vehicle_tracker[matched_id]['speed']:.2f} km/h"
                    cv2.putText(frame, speed_text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    # Print to console
                    print(f"Vehicle ID: {matched_id}, Class: {model.names[cls]}, Confidence: {conf:.2f}, Speed: {vehicle_tracker[matched_id]['speed']:.2f} km/h")
                else:
                    # Print to console for new vehicles without speed
                    print(f"Vehicle ID: {next_vehicle_id - 1}, Class: {model.names[cls]}, Confidence: {conf:.2f}, Speed: Not yet calculated")

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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