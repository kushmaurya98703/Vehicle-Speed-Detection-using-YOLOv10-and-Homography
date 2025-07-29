import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO

# -------------------- CONFIGURATION --------------------
MAX_CAMERA_INDEX = 5       # Maximum camera indices to probe
FRAME_WIDTH = 1920         # Desired capture resolution width
FRAME_HEIGHT = 1080        # Desired capture resolution height
FRAME_RATE = 30            # Desired FPS
SPEED_HISTORY_SIZE = 30    # Frames for moving average
MIN_TIME_DIFF = 0.01       # Avoid division by zero
MAX_SPEED_KMPH = 200.0     # Filter unrealistic speeds
OUTPUT_PATH = 'output_video_with_speed.mp4'

# -------------------- GLOBAL STATE --------------------
points = []  # For homography selection
vehicle_tracker = {}  # Active tracked vehicles
next_vehicle_id = 0   # Incremental ID

# -------------------- Helper Functions --------------------
def get_working_cameras(max_index=MAX_CAMERA_INDEX):
    working = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                working.append(i)
            cap.release()
    return working


def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append([x, y])
        cv2.circle(param['frame'], (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Select 4 Points', param['frame'])
        print(f"Point {len(points)}: [{x}, {y}]")
        if len(points) == 4:
            print("\nCollected 4 src_points for homography:")
            for i, pt in enumerate(points, 1):
                print(f"  Point {i}: {pt[0]}, {pt[1]}")
            print("\nPlease supply corresponding real-world dst_points.")


def transform_point(point, H):
    pt = np.array([point[0], point[1], 1.0], dtype=np.float32)
    res = H.dot(pt)
    return (res[0]/res[2], res[1]/res[2])


def calculate_distance(p1, p2):
    return np.linalg.norm((p1[0]-p2[0], p1[1]-p2[1]))


def calculate_moving_average(history):
    return sum(history)/len(history) if history else None

# -------------------- MAIN --------------------
def main():
    global next_vehicle_id

    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("Running on CPU. To enable GPU, install a CUDA-enabled PyTorch build.")
    else:
        print(f"CUDA version: {torch.version.cuda}")

    model = YOLO('yolov10n.pt')
    if device == 'cuda':
        model.to(device)

    cap = cv2.VideoCapture('video-1.mp4')


    print("Press 'p' to pause and select 4 homography points...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame for homography.")
            return
        cv2.imshow('Press "p" to pause', frame)
        if cv2.waitKey(1) & 0xFF == ord('p'):
            frame_copy = frame.copy()
            break

    cv2.namedWindow('Select 4 Points', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Select 4 Points', mouse_callback, {'frame': frame_copy})
    while len(points) < 4:
        cv2.imshow('Select 4 Points', frame_copy)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Aborting homography setup.")
            return
    cv2.destroyWindow('Select 4 Points')

    src_pts = np.float32(points)
    dst_pts = np.float32(
        [[0,0],
         [13,0],
         [0,35],
         [13,35]])
    H, _ = cv2.findHomography(src_pts, dst_pts)

    height, width = frame_copy.shape[:2]
    # Select codec based on file extension
    ext = OUTPUT_PATH.split('.')[-1].lower()
    if ext in ('avi', 'mpg'):
        codec = 'XVID'
    else:
        codec = 'mp4v'
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, FRAME_RATE, (width, height))
    if not out.isOpened():
        print(f" VideoWriter failed to open using codec '{codec}'.")
        return

    print("Press 's' to start detection...")
    while True:
        ret, frame = cap.read()
        if not ret:
            return
        cv2.imshow('Press "s" to start', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small = cv2.resize(frame, (416, 416))
        results = model(small, conf=0.45, iou=0.45, device=device)
        detections = {}

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                scale_x, scale_y = FRAME_WIDTH/416, FRAME_HEIGHT/416
                x1, x2 = int(x1*scale_x), int(x2*scale_x)
                y1, y2 = int(y1*scale_y), int(y2*scale_y)
                cls = int(box.cls[0])
                center = ((x1+x2)//2, (y1+y2)//2)
                if cls in [2,3,5,7,8]:
                    world_pt = transform_point(center, H)
                    inside = cv2.pointPolygonTest(src_pts, center, False) >= 0

                    matched, min_d = -1, float('inf')
                    for vid, data in vehicle_tracker.items():
                        d = calculate_distance(center, data['last_pixel_position'])
                        if d < min_d and d < 100:
                            min_d, matched = d, vid

                    if inside:
                        if matched != -1:
                            data = vehicle_tracker[matched]
                            dt = time.time() - data['last_frame_time']
                            if dt > MIN_TIME_DIFF:
                                dist_m = calculate_distance(world_pt, data['last_position_world'])
                                speed = (dist_m/dt)*3.6
                                if speed <= MAX_SPEED_KMPH:
                                    data['speed_history'].append(speed)
                                    data['speed_history'] = data['speed_history'][-SPEED_HISTORY_SIZE:]
                                    data['speed'] = calculate_moving_average(data['speed_history'])
                            data.update({
                                'last_position_world': world_pt,
                                'last_pixel_position': center,
                                'last_frame_time': time.time()
                            })
                            detections[matched] = data
                        else:
                            vehicle_tracker[next_vehicle_id] = {
                                'last_position_world': world_pt,
                                'last_pixel_position': center,
                                'last_frame_time': time.time(),
                                'speed': None,
                                'speed_history': []
                            }
                            detections[next_vehicle_id] = vehicle_tracker[next_vehicle_id]
                            next_vehicle_id += 1

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                        vid_to_show = matched if matched != -1 else next_vehicle_id-1
                        if vehicle_tracker.get(vid_to_show, {}).get('speed') is not None:
                            speed_txt = vehicle_tracker[vid_to_show]['speed']
                            cv2.putText(frame, f"ID:{vid_to_show} {speed_txt:.1f} km/h", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        for vid in list(vehicle_tracker):
            if vid not in detections:
                del vehicle_tracker[vid]

        if frame.dtype == np.uint8 and frame.shape[2] == 3:
            out.write(frame)
        cv2.namedWindow('YOLOv10 Vehicle Detection with Speed', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('YOLOv10 Vehicle Detection with Speed', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('YOLOv10 Vehicle Detection with Speed', frame)


        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            cap.release()
            cam_idx = (cam_idx + 1) % len(camera_indices)
            cap = cv2.VideoCapture(camera_indices[cam_idx])
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Done. Output saved to {OUTPUT_PATH}")

if __name__ == '__main__':
    main()
