import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict
import math

# =====================================
# SETTINGS
# =====================================

VIDEO_PATH = "0212.mp4"
MODEL_PATH = r"D:\practice research\drone_Data\T intersection fixed drone\best.pt"  # Changed to your trained model
CONFIDENCE = 0.7
IMAGE_SIZE = 1280

VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle"]

# =====================================
# LOAD MODEL
# =====================================

model = YOLO(MODEL_PATH)

# =====================================
# STEP 1: DRAW ROI POLYGON
# Left Click = Add Point
# Right Click = Finish
# =====================================

roi_points = []
drawing_done = False

def draw_roi(event, x, y, flags, param):
    global roi_points, drawing_done
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        drawing_done = True

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
if not ret:
    print("Error loading video")
    exit()

cv2.namedWindow("Draw ROI", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Draw ROI", 1000, 700)
cv2.setMouseCallback("Draw ROI", draw_roi)

while True:
    temp = frame.copy()

    for p in roi_points:
        cv2.circle(temp, p, 5, (0, 0, 255), -1)

    if len(roi_points) > 1:
        cv2.polylines(temp, [np.array(roi_points)], False, (255, 0, 0), 2)

    cv2.imshow("Draw ROI", temp)

    if drawing_done and len(roi_points) > 2:
        break

    if cv2.waitKey(1) & 0xFF == 27:
        exit()

cv2.destroyWindow("Draw ROI")
roi_polygon = np.array(roi_points)

# =====================================
# STEP 2: DRAW CALIBRATION LINE
# Click 2 points
# =====================================

calibration_points = []

def draw_line(event, x, y, flags, param):
    global calibration_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(calibration_points) < 2:
            calibration_points.append((x, y))

cv2.namedWindow("Draw Calibration Line", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Draw Calibration Line", 1000, 700)
cv2.setMouseCallback("Draw Calibration Line", draw_line)

while True:
    temp = frame.copy()

    for p in calibration_points:
        cv2.circle(temp, p, 6, (0, 255, 0), -1)

    if len(calibration_points) == 2:
        cv2.line(temp, calibration_points[0], calibration_points[1], (0, 255, 0), 2)

    cv2.imshow("Draw Calibration Line", temp)

    if len(calibration_points) == 2:
        break

    if cv2.waitKey(1) & 0xFF == 27:
        exit()

cv2.destroyWindow("Draw Calibration Line")

real_distance = float(input("Enter real-world distance (meters): "))

(x1, y1), (x2, y2) = calibration_points
pixel_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
meters_per_pixel = real_distance / pixel_distance

print(f"Calibration completed: {meters_per_pixel:.6f} meters per pixel")

# =====================================
# STEP 3: PROCESS VIDEO
# =====================================

cap.release()
cap = cv2.VideoCapture(VIDEO_PATH)

track_positions = {}
track_roi_status = {}

per_second_data = defaultdict(lambda: {
    "total": 0,
    "car": 0,
    "truck": 0,
    "bus": 0,
    "motorcycle": 0,
    "entered": 0,
    "exited": 0,
    "speeds": []
})

print("Processing video...")

cv2.namedWindow("Drone Traffic Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Drone Traffic Detection", 1200, 800)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    current_sec = int(frame_time)

    results = model.track(
        frame,
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        persist=True,
        tracker="bytetrack.yaml"
    )

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        for box, track_id, cls in zip(boxes, ids, classes):

            class_name = model.names[int(cls)]
            if class_name not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            inside = cv2.pointPolygonTest(roi_polygon, (cx, cy), False) >= 0

            # ENTRY / EXIT detection
            if track_id not in track_roi_status:
                track_roi_status[track_id] = inside
            else:
                prev_status = track_roi_status[track_id]

                if not prev_status and inside:
                    per_second_data[current_sec]["entered"] += 1

                if prev_status and not inside:
                    per_second_data[current_sec]["exited"] += 1

                track_roi_status[track_id] = inside

            # Count & speed only if inside
            if inside:
                per_second_data[current_sec]["total"] += 1
                per_second_data[current_sec][class_name] += 1

                if track_id in track_positions:
                    prev_cx, prev_cy, prev_time = track_positions[track_id]

                    pixel_move = math.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                    time_diff = frame_time - prev_time

                    if time_diff > 0:
                        distance_m = pixel_move * meters_per_pixel
                        speed_m_s = distance_m / time_diff
                        speed_km_h = speed_m_s * 3.6

                        if 0 < speed_km_h < 150:
                            per_second_data[current_sec]["speeds"].append(speed_km_h)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                cv2.putText(frame, f"{class_name}-{int(track_id)}",
                            (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            track_positions[track_id] = (cx, cy, frame_time)

    # Draw ROI
    cv2.polylines(frame, [roi_polygon], True, (255, 0, 0), 2)

    cv2.putText(frame, f"Time: {current_sec} sec",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Drone Traffic Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# =====================================
# STEP 4: SAVE RESULTS
# =====================================

final_data = []

for sec in sorted(per_second_data.keys()):
    data = per_second_data[sec]
    avg_speed = np.mean(data["speeds"]) if len(data["speeds"]) > 0 else 0

    final_data.append({
        "Timestamp_sec": sec,
        "Total_Inside_ROI": data["total"],
        "Car": data["car"],
        "Truck": data["truck"],
        "Bus": data["bus"],
        "Motorcycle": data["motorcycle"],
        "Vehicles_Entered_ROI": data["entered"],
        "Vehicles_Exited_ROI": data["exited"],
        "Avg_Speed_km_h": round(avg_speed, 2)
    })

df = pd.DataFrame(final_data)
df.to_excel("drone_traffic_calibrated_results.xlsx", index=False)

print("Processing complete.")
print("Excel saved as drone_traffic_calibrated_results.xlsx")