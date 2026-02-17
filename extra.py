import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict
import math
from datetime import datetime
import os

# =====================================
# SETTINGS
# =====================================

VIDEO_PATH = "0212.mp4"
MODEL_PATH = r"D:\practice research\drone_Data\T intersection fixed drone\best.pt"
CONFIDENCE = 0.7
IMAGE_SIZE = 1280  # YOLO input size (will resize 4K frames)

VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle"]

# Road parameters - ADJUST THESE BASED ON YOUR SITE
ROAD_LENGTH_M = 50  # meters (length of your ROI segment)
NUM_LANES = 4       # number of lanes in the road
ROAD_LENGTH_KM = ROAD_LENGTH_M / 1000

# Processing optimization for 4K@60
PROCESS_EVERY_N_FRAMES = 2  # Process every 2nd frame (effective 30 fps from 60 fps)

# Video output settings
SAVE_ANNOTATED_VIDEO = True  # Set to True to save annotated video
OUTPUT_VIDEO_FPS = 30  # Output video FPS (will be adjusted based on processing)
OUTPUT_VIDEO_CODEC = 'mp4v'  # Codec for output video

# =====================================
# CHECK VIDEO PROPERTIES
# =====================================

print("=" * 60)
print("DRONE TRAFFIC ANALYSIS WITH VIDEO RECORDING")
print("=" * 60)

print("\nChecking video properties...")
cap_check = cv2.VideoCapture(VIDEO_PATH)
FPS = cap_check.get(cv2.CAP_PROP_FPS)
frame_width = int(cap_check.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap_check.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
cap_check.release()

print(f"\n📹 Video Properties:")
print(f"  - FPS: {FPS}")
print(f"  - Resolution: {frame_width} x {frame_height}")
print(f"  - Total Frames: {total_frames}")
print(f"  - Duration: {total_frames/FPS:.2f} seconds")
print(f"\n⚙️ Processing Settings:")
print(f"  - Processing every {PROCESS_EVERY_N_FRAMES} frame(s)")
print(f"  - Effective processing FPS: {FPS/PROCESS_EVERY_N_FRAMES:.1f}")
print(f"  - Save annotated video: {SAVE_ANNOTATED_VIDEO}")

# =====================================
# LOAD MODEL
# =====================================

print("\n📥 Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("✅ Model loaded successfully!")

# =====================================
# STEP 1: DRAW ROI POLYGON
# =====================================

roi_points = []
drawing_done = False

def draw_roi(event, x, y, flags, param):
    global roi_points, drawing_done
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points.append((x, y))
        print(f"  Point added: ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN:
        drawing_done = True
        print("  ROI drawing completed")

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
if not ret:
    print("❌ Error loading video")
    exit()

# Resize frame for display if 4K
display_frame = cv2.resize(frame, (1280, 720)) if frame_width > 1920 else frame

cv2.namedWindow("Draw ROI", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Draw ROI", 1280, 720)
cv2.setMouseCallback("Draw ROI", draw_roi)

print("\n🎯 STEP 1: DRAW ROI POLYGON")
print("-" * 40)
print("  Left Click: Add points")
print("  Right Click: Finish polygon")
print("  Press ESC to exit")

while True:
    temp = display_frame.copy()
    
    # Draw points
    for p in roi_points:
        # Scale points back for display if we resized
        display_p = (int(p[0] * display_frame.shape[1] / frame.shape[1]), 
                    int(p[1] * display_frame.shape[0] / frame.shape[0])) if frame_width > 1920 else p
        cv2.circle(temp, display_p, 5, (0, 0, 255), -1)

    if len(roi_points) > 1:
        # Scale polygon for display
        if frame_width > 1920:
            scaled_points = [(int(x * display_frame.shape[1] / frame.shape[1]), 
                            int(y * display_frame.shape[0] / frame.shape[0])) for x, y in roi_points]
            cv2.polylines(temp, [np.array(scaled_points)], False, (255, 0, 0), 2)
        else:
            cv2.polylines(temp, [np.array(roi_points)], False, (255, 0, 0), 2)

    cv2.imshow("Draw ROI", temp)

    if drawing_done and len(roi_points) > 2:
        break

    if cv2.waitKey(1) & 0xFF == 27:
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyWindow("Draw ROI")
roi_polygon = np.array(roi_points)
print(f"✅ ROI polygon created with {len(roi_points)} points")

# =====================================
# STEP 2: DRAW CALIBRATION LINE
# =====================================

calibration_points = []

def draw_line(event, x, y, flags, param):
    global calibration_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(calibration_points) < 2:
            calibration_points.append((x, y))
            print(f"  Calibration point {len(calibration_points)}: ({x}, {y})")

cv2.namedWindow("Draw Calibration Line", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Draw Calibration Line", 1280, 720)
cv2.setMouseCallback("Draw Calibration Line", draw_line)

print("\n📏 STEP 2: DRAW CALIBRATION LINE")
print("-" * 40)
print("  Click two points on the road with known distance")
print("  Press ESC to exit")

while True:
    temp = display_frame.copy()
    
    for p in calibration_points:
        cv2.circle(temp, p, 6, (0, 255, 0), -1)

    if len(calibration_points) == 2:
        cv2.line(temp, calibration_points[0], calibration_points[1], (0, 255, 0), 2)

    cv2.imshow("Draw Calibration Line", temp)

    if len(calibration_points) == 2:
        break

    if cv2.waitKey(1) & 0xFF == 27:
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyWindow("Draw Calibration Line")

# Get real-world distance
real_distance = float(input("\n📝 Enter real-world distance between points (meters): "))

# Calculate meters per pixel
if frame_width > 1920:
    # Scale points back to original resolution
    x1 = int(calibration_points[0][0] * frame.shape[1] / display_frame.shape[1])
    y1 = int(calibration_points[0][1] * frame.shape[0] / display_frame.shape[0])
    x2 = int(calibration_points[1][0] * frame.shape[1] / display_frame.shape[1])
    y2 = int(calibration_points[1][1] * frame.shape[0] / display_frame.shape[0])
else:
    x1, y1 = calibration_points[0]
    x2, y2 = calibration_points[1]

pixel_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
meters_per_pixel = real_distance / pixel_distance

print(f"\n✅ Calibration completed:")
print(f"  - Pixel distance: {pixel_distance:.2f} pixels")
print(f"  - Real distance: {real_distance} meters")
print(f"  - Meters per pixel: {meters_per_pixel:.6f}")

# =====================================
# STEP 3: SETUP VIDEO WRITER
# =====================================

if SAVE_ANNOTATED_VIDEO:
    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_path = f"annotated_traffic_{timestamp}.mp4"
    
    # Calculate output video dimensions (maintain aspect ratio for display)
    output_width = 1280
    output_height = int(output_width * frame_height / frame_width)
    if output_height > 720:
        output_height = 720
        output_width = int(output_height * frame_width / frame_height)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*OUTPUT_VIDEO_CODEC)
    out_video = cv2.VideoWriter(output_video_path, fourcc, OUTPUT_VIDEO_FPS, (output_width, output_height))
    
    print(f"\n🎥 Video recording enabled:")
    print(f"  - Output file: {output_video_path}")
    print(f"  - Output FPS: {OUTPUT_VIDEO_FPS}")
    print(f"  - Output resolution: {output_width} x {output_height}")

# =====================================
# STEP 4: PROCESS VIDEO
# =====================================

cap.release()
cap = cv2.VideoCapture(VIDEO_PATH)

# Tracking data structures
track_positions = {}
track_roi_status = {}
vehicle_first_seen = {}
vehicle_speeds = defaultdict(list)  # Store speed history for each vehicle

# Store unique vehicles per second
unique_vehicles_per_second = defaultdict(set)
vehicle_counted_this_second = defaultdict(lambda: defaultdict(bool))

# Per second statistics
per_second_data = defaultdict(lambda: {
    "car": 0,
    "truck": 0,
    "bus": 0,
    "motorcycle": 0,
    "entered": 0,
    "exited": 0,
    "speeds": [],
    "unique_vehicles": set()
})

print("\n🚀 STEP 3: PROCESSING VIDEO")
print("-" * 40)
print("Press ESC to stop early")
print("Processing...")

cv2.namedWindow("Drone Traffic Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Drone Traffic Detection", 1280, 720)

frame_count = 0
processed_frames = 0
start_time = datetime.now()

# For progress tracking
last_progress_update = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    
    # Skip frames for processing efficiency
    if frame_count % PROCESS_EVERY_N_FRAMES != 0:
        continue
    
    processed_frames += 1
    
    # Calculate accurate timestamp
    frame_time = frame_count / FPS
    current_sec = int(frame_time)

    # Run YOLO tracking - FIXED: changed imsz to imgsz
    results = model.track(
        frame,
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,  # <-- CORRECTED PARAMETER NAME
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    # Create annotated frame
    annotated_frame = frame.copy()

    # Process detections
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy() if results[0].boxes.conf is not None else None

        for i, (box, track_id, cls) in enumerate(zip(boxes, ids, classes)):
            class_name = model.names[int(cls)]
            conf = confidences[i] if confidences is not None else 0
            
            # Filter vehicle classes
            if class_name not in VEHICLE_CLASSES:
                continue

            # Get bounding box and center point
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Check if vehicle is inside ROI
            inside = cv2.pointPolygonTest(roi_polygon, (cx, cy), False) >= 0

            # ENTRY / EXIT detection
            if track_id not in track_roi_status:
                track_roi_status[track_id] = inside
                if inside:
                    vehicle_first_seen[track_id] = frame_time
            else:
                prev_status = track_roi_status[track_id]

                if not prev_status and inside:
                    per_second_data[current_sec]["entered"] += 1
                    vehicle_first_seen[track_id] = frame_time

                if prev_status and not inside:
                    per_second_data[current_sec]["exited"] += 1

                track_roi_status[track_id] = inside

            # Process vehicles inside ROI
            current_speed = 0
            if inside:
                # Add to unique vehicles set for this second
                per_second_data[current_sec]["unique_vehicles"].add(track_id)
                
                # Count vehicle type only once per second
                if not vehicle_counted_this_second[track_id][current_sec]:
                    per_second_data[current_sec][class_name] += 1
                    vehicle_counted_this_second[track_id][current_sec] = True

                # Speed calculation
                if track_id in track_positions:
                    prev_cx, prev_cy, prev_time = track_positions[track_id]

                    pixel_move = math.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                    time_diff = frame_time - prev_time

                    if time_diff > 0:
                        distance_m = pixel_move * meters_per_pixel
                        speed_m_s = distance_m / time_diff
                        speed_km_h = speed_m_s * 3.6

                        # Validate speed
                        if 0 < speed_km_h < 150:
                            per_second_data[current_sec]["speeds"].append(speed_km_h)
                            vehicle_speeds[track_id].append(speed_km_h)
                            current_speed = speed_km_h

                # Draw bounding box with color based on vehicle type
                color_map = {
                    "car": (0, 255, 0),        # Green
                    "truck": (255, 165, 0),     # Orange
                    "bus": (255, 0, 0),         # Blue
                    "motorcycle": (255, 255, 0)  # Cyan
                }
                color = color_map.get(class_name, (0, 255, 0))
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Create label with vehicle info
                label = f"{class_name}-{int(track_id)}"
                if current_speed > 0:
                    label += f" {current_speed:.1f}km/h"
                
                # Draw label background
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(annotated_frame, 
                            (int(x1), int(y1) - label_height - 10),
                            (int(x1) + label_width, int(y1) - 5),
                            color, -1)
                
                # Draw label text
                cv2.putText(annotated_frame, label, (int(x1), int(y1)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                # Draw center point
                cv2.circle(annotated_frame, (cx, cy), 4, (0, 0, 255), -1)

            # Update position tracking
            track_positions[track_id] = (cx, cy, frame_time)

    # Draw ROI polygon
    cv2.polylines(annotated_frame, [roi_polygon], True, (255, 0, 0), 3)
    
    # Calculate current statistics
    unique_count = len(per_second_data[current_sec]["unique_vehicles"])
    avg_speed = np.mean(per_second_data[current_sec]["speeds"]) if per_second_data[current_sec]["speeds"] else 0
    density = (unique_count / ROAD_LENGTH_KM) / NUM_LANES if ROAD_LENGTH_KM > 0 else 0
    
    # Create info panel
    info_panel = np.zeros((150, annotated_frame.shape[1], 3), dtype=np.uint8)
    
    # Add text to info panel
    cv2.putText(info_panel, f"Time: {current_sec}s", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(info_panel, f"Vehicles: {unique_count}", (200, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(info_panel, f"Density: {density:.1f} veh/km/lane", (400, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(info_panel, f"Speed: {avg_speed:.1f} km/h", (700, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Add vehicle counts by type
    y_offset = 70
    cv2.putText(info_panel, f"Cars: {per_second_data[current_sec]['car']}", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(info_panel, f"Trucks: {per_second_data[current_sec]['truck']}", (150, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
    cv2.putText(info_panel, f"Buses: {per_second_data[current_sec]['bus']}", (290, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(info_panel, f"Motorcycles: {per_second_data[current_sec]['motorcycle']}", (430, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(info_panel, f"Entered: {per_second_data[current_sec]['entered']}", (600, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(info_panel, f"Exited: {per_second_data[current_sec]['exited']}", (750, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Add progress bar
    progress = frame_count / total_frames
    bar_width = int(progress * (annotated_frame.shape[1] - 20))
    cv2.rectangle(info_panel, (10, 120), (annotated_frame.shape[1] - 10, 140), (100, 100, 100), -1)
    cv2.rectangle(info_panel, (10, 120), (10 + bar_width, 140), (0, 255, 0), -1)
    cv2.putText(info_panel, f"Progress: {progress*100:.1f}%", (annotated_frame.shape[1] - 200, 135), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Combine frame and info panel
    display_frame = np.vstack([annotated_frame, info_panel])
    
    # Resize for display if needed
    if display_frame.shape[1] > 1280:
        scale = 1280 / display_frame.shape[1]
        new_width = 1280
        new_height = int(display_frame.shape[0] * scale)
        display_frame = cv2.resize(display_frame, (new_width, new_height))

    # Show frame
    cv2.imshow("Drone Traffic Detection", display_frame)

    # Save annotated frame to video
    if SAVE_ANNOTATED_VIDEO:
        # Resize to output dimensions
        output_frame = cv2.resize(display_frame, (output_width, output_height))
        out_video.write(output_frame)

    # Progress update every 5 seconds
    if current_sec > last_progress_update:
        elapsed = (datetime.now() - start_time).total_seconds()
        fps_processing = processed_frames / elapsed if elapsed > 0 else 0
        print(f"  Time: {current_sec}s | Vehicles: {unique_count} | "
              f"Density: {density:.1f} | Speed: {avg_speed:.1f} km/h | "
              f"Processing: {fps_processing:.1f} fps")
        last_progress_update = current_sec

    if cv2.waitKey(1) & 0xFF == 27:
        print("\n⏹️ Processing stopped by user")
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
if SAVE_ANNOTATED_VIDEO:
    out_video.release()

# Calculate processing statistics
elapsed_total = (datetime.now() - start_time).total_seconds()
avg_processing_fps = processed_frames / elapsed_total if elapsed_total > 0 else 0

print(f"\n✅ Processing complete!")
print(f"  - Total frames: {frame_count}")
print(f"  - Processed frames: {processed_frames}")
print(f"  - Processing time: {elapsed_total:.1f} seconds")
print(f"  - Average processing FPS: {avg_processing_fps:.1f}")
print(f"  - Video duration: {frame_count/FPS:.2f} seconds")

# =====================================
# STEP 5: CALCULATE DENSITY AND SAVE RESULTS
# =====================================

print("\n📊 STEP 4: CALCULATING STATISTICS")
print("-" * 40)

final_data = []

for sec in sorted(per_second_data.keys()):
    data = per_second_data[sec]
    
    # Number of unique vehicles in ROI this second
    unique_vehicles = len(data["unique_vehicles"])
    
    # Calculate average speed
    avg_speed = np.mean(data["speeds"]) if len(data["speeds"]) > 0 else 0
    
    # Calculate density
    vehicles_per_km = unique_vehicles / ROAD_LENGTH_KM if ROAD_LENGTH_KM > 0 else 0
    density_per_lane = vehicles_per_km / NUM_LANES if NUM_LANES > 0 else 0
    
    final_data.append({
        "Timestamp_sec": sec,
        "Unique_Vehicles_in_ROI": unique_vehicles,
        "Vehicles_per_km": round(vehicles_per_km, 2),
        "Density_per_lane_veh_per_km": round(density_per_lane, 2),
        "Car": data["car"],
        "Truck": data["truck"],
        "Bus": data["bus"],
        "Motorcycle": data["motorcycle"],
        "Vehicles_Entered_ROI": data["entered"],
        "Vehicles_Exited_ROI": data["exited"],
        "Avg_Speed_km_h": round(avg_speed, 2),
        "Speed_Samples": len(data["speeds"])
    })

# Create DataFrame
df = pd.DataFrame(final_data)

# Calculate summary statistics
print("\n📈 SUMMARY STATISTICS")
print("=" * 60)
print(f"Video duration: {len(final_data)} seconds")
print(f"Road segment length: {ROAD_LENGTH_M} meters ({ROAD_LENGTH_KM} km)")
print(f"Number of lanes: {NUM_LANES}")
print(f"\n🚦 Traffic Statistics:")
print(f"  - Avg unique vehicles/second: {df['Unique_Vehicles_in_ROI'].mean():.1f}")
print(f"  - Max unique vehicles/second: {df['Unique_Vehicles_in_ROI'].max()}")
print(f"  - Min unique vehicles/second: {df['Unique_Vehicles_in_ROI'].min()}")
print(f"\n📏 Density Statistics (veh/km/lane):")
print(f"  - Average density: {df['Density_per_lane_veh_per_km'].mean():.1f}")
print(f"  - Max density: {df['Density_per_lane_veh_per_km'].max():.1f}")
print(f"  - Min density: {df['Density_per_lane_veh_per_km'].min():.1f}")
print(f"\n⚡ Speed Statistics:")
print(f"  - Average speed: {df['Avg_Speed_km_h'].mean():.1f} km/h")
print(f"  - Max speed: {df['Avg_Speed_km_h'].max():.1f} km/h")
print(f"  - Min speed: {df['Avg_Speed_km_h'].min():.1f} km/h")
print(f"\n🚗 Vehicle Counts:")
print(f"  - Total Cars: {df['Car'].sum()}")
print(f"  - Total Trucks: {df['Truck'].sum()}")
print(f"  - Total Buses: {df['Bus'].sum()}")
print(f"  - Total Motorcycles: {df['Motorcycle'].sum()}")
print(f"  - Total Vehicles Entered: {df['Vehicles_Entered_ROI'].sum()}")
print(f"  - Total Vehicles Exited: {df['Vehicles_Exited_ROI'].sum()}")

# Save to Excel
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
excel_filename = f"traffic_analysis_{timestamp}.xlsx"

# Create Excel with multiple sheets
with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='Per_Second_Data', index=False)
    
    # Summary sheet
    summary_data = {
        'Metric': ['Video FPS', 'Processed FPS', 'Duration (seconds)', 'Road Length (m)', 'Number of Lanes',
                   'Avg Vehicles/Second', 'Max Vehicles/Second', 'Avg Density (veh/km/lane)', 
                   'Max Density (veh/km/lane)', 'Avg Speed (km/h)', 'Max Speed (km/h)', 'Min Speed (km/h)',
                   'Total Cars', 'Total Trucks', 'Total Buses', 'Total Motorcycles', 
                   'Total Entries', 'Total Exits'],
        'Value': [FPS, FPS/PROCESS_EVERY_N_FRAMES, len(final_data), ROAD_LENGTH_M, NUM_LANES,
                  round(df['Unique_Vehicles_in_ROI'].mean(), 2), df['Unique_Vehicles_in_ROI'].max(),
                  round(df['Density_per_lane_veh_per_km'].mean(), 2), 
                  round(df['Density_per_lane_veh_per_km'].max(), 2),
                  round(df['Avg_Speed_km_h'].mean(), 2),
                  round(df['Avg_Speed_km_h'].max(), 2),
                  round(df['Avg_Speed_km_h'].min(), 2),
                  df['Car'].sum(), df['Truck'].sum(),
                  df['Bus'].sum(), df['Motorcycle'].sum(), 
                  df['Vehicles_Entered_ROI'].sum(), df['Vehicles_Exited_ROI'].sum()]
    }
    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

print(f"\n✅ Excel file saved: {excel_filename}")

if SAVE_ANNOTATED_VIDEO:
    print(f"✅ Annotated video saved: {output_video_path}")

# Display first few rows
print("\n📋 FIRST 10 ROWS OF DATA")
print("=" * 60)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(df.head(10).to_string())

# Optional: Create visualization
try:
    import matplotlib.pyplot as plt
    
    print("\n📊 Generating plots...")
    
    # Create a figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    # Plot 1: Vehicle count over time
    axes[0].plot(df['Timestamp_sec'], df['Unique_Vehicles_in_ROI'], 'b-', linewidth=1, alpha=0.7)
    axes[0].fill_between(df['Timestamp_sec'], df['Unique_Vehicles_in_ROI'], alpha=0.3)
    axes[0].set_ylabel('Vehicles in ROI', fontsize=10)
    axes[0].set_title('Vehicle Count Over Time', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([df['Timestamp_sec'].min(), df['Timestamp_sec'].max()])
    
    # Plot 2: Density over time
    axes[1].plot(df['Timestamp_sec'], df['Density_per_lane_veh_per_km'], 'r-', linewidth=1, alpha=0.7)
    axes[1].fill_between(df['Timestamp_sec'], df['Density_per_lane_veh_per_km'], alpha=0.3, color='red')
    axes[1].set_ylabel('Density (veh/km/lane)', fontsize=10)
    axes[1].set_title('Traffic Density Over Time', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([df['Timestamp_sec'].min(), df['Timestamp_sec'].max()])
    
    # Plot 3: Speed over time
    axes[2].plot(df['Timestamp_sec'], df['Avg_Speed_km_h'], 'g-', linewidth=1, alpha=0.7)
    axes[2].fill_between(df['Timestamp_sec'], df['Avg_Speed_km_h'], alpha=0.3, color='green')
    axes[2].set_ylabel('Speed (km/h)', fontsize=10)
    axes[2].set_title('Average Speed Over Time', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim([df['Timestamp_sec'].min(), df['Timestamp_sec'].max()])
    
    # Plot 4: Vehicle type distribution over time (stacked area)
    axes[3].stackplot(df['Timestamp_sec'], df['Car'], df['Truck'], df['Bus'], df['Motorcycle'],
                      labels=['Car', 'Truck', 'Bus', 'Motorcycle'],
                      colors=['green', 'orange', 'blue', 'cyan'], alpha=0.7)
    axes[3].set_ylabel('Vehicle Count', fontsize=10)
    axes[3].set_xlabel('Time (seconds)', fontsize=10)
    axes[3].set_title('Vehicle Type Distribution Over Time', fontsize=12, fontweight='bold')
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xlim([df['Timestamp_sec'].min(), df['Timestamp_sec'].max()])
    
    plt.tight_layout()
    plot_filename = f"traffic_analysis_plot_{timestamp}.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Plot saved: {plot_filename}")
    
except ImportImportError:
    print("\n⚠️ Matplotlib not installed - skipping visualization")
    print("  Install with: pip install matplotlib")

print("\n" + "=" * 60)
print("🎉 PROCESSING COMPLETE! 🎉")
print("=" * 60)
print(f"\nOutput files:")
print(f"  📊 Excel data: {excel_filename}")
if SAVE_ANNOTATED_VIDEO:
    print(f"  🎥 Annotated video: {output_video_path}")
print("\n" + "=" * 60)