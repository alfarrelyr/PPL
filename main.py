import cv2
import os
import time
import math
from collections import deque

from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "yolo11n.pt"

# 0 = webcam
# Atau ganti dengan:
# "videos/test.mp4"
SOURCE = 0

OUTPUT_DIR = "incidents"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Classes from COCO
# 2  = car
# 3  = motorcycle
# 5  = bus
# 7  = truck
# 0  = person
VEHICLE_CLASSES = {2, 3, 5, 7}

# Detection confidence
CONFIDENCE = 0.35

# Minimum IoU / overlap to consider possible collision
COLLISION_IOU = 0.10

# Minimum sudden speed change
MIN_SPEED_CHANGE = 12.0

# How many frames the accident warning remains active
ACCIDENT_HOLD_FRAMES = 40

# Frames stored before accident
PRE_EVENT_SECONDS = 5

# FPS fallback for webcam
DEFAULT_FPS = 30


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def center_of_box(box):
    x1, y1, x2, y2 = box
    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0
    )


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def calculate_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    intersection = iw * ih

    if intersection <= 0:
        return 0.0

    area_a = box_area(box_a)
    area_b = box_area(box_b)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def distance(point_a, point_b):
    return math.sqrt(
        (point_a[0] - point_b[0]) ** 2 +
        (point_a[1] - point_b[1]) ** 2
    )


def draw_label(frame, text, position, color):
    x, y = position

    (w, h), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        2
    )

    cv2.rectangle(
        frame,
        (x, y - h - baseline - 5),
        (x + w + 8, y + 5),
        color,
        -1
    )

    cv2.putText(
        frame,
        text,
        (x + 4, y - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


# ============================================================
# TRACKING DATA
# ============================================================

# Previous center position of each tracked object
previous_positions = {}

# Previous speed of each tracked object
previous_speeds = {}

# Current speed of each object
current_speeds = {}

# Accident state
accident_counter = 0
accident_active = False

# Prevent saving a new incident every frame
last_incident_time = 0

# Minimum interval between incident recordings
INCIDENT_COOLDOWN = 10


# ============================================================
# VIDEO SOURCE
# ============================================================

cap = cv2.VideoCapture(SOURCE)

if not cap.isOpened():
    raise RuntimeError(
        f"Tidak dapat membuka source video: {SOURCE}"
    )


fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 1 or math.isnan(fps):
    fps = DEFAULT_FPS

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if width <= 0:
    width = 1280

if height <= 0:
    height = 720


# ============================================================
# YOLO MODEL
# ============================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("Model berhasil dimuat.")
print("Tekan Q untuk keluar.")


# ============================================================
# VIDEO BUFFER
# ============================================================

buffer_size = int(fps * PRE_EVENT_SECONDS)

frame_buffer = deque(maxlen=buffer_size)


# ============================================================
# INCIDENT VIDEO WRITER
# ============================================================

incident_writer = None
incident_remaining_frames = 0


def start_incident_recording():
    global incident_writer
    global incident_remaining_frames
    global last_incident_time

    now = time.time()

    if now - last_incident_time < INCIDENT_COOLDOWN:
        return

    last_incident_time = now

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    filename = os.path.join(
        OUTPUT_DIR,
        f"accident_{timestamp}.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    incident_writer = cv2.VideoWriter(
        filename,
        fourcc,
        fps,
        (width, height)
    )

    # Write buffered frames first
    for buffered_frame in frame_buffer:
        incident_writer.write(buffered_frame)

    # Continue recording after accident
    incident_remaining_frames = int(fps * 5)

    print()
    print("=" * 60)
    print("ACCIDENT TERDETEKSI!")
    print(f"Video disimpan: {filename}")
    print("=" * 60)
    print()


def finish_incident_recording():
    global incident_writer

    if incident_writer is not None:
        incident_writer.release()
        incident_writer = None


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video selesai atau frame tidak dapat dibaca.")
        break

    # Keep original frame in buffer
    frame_buffer.append(frame.copy())

    # --------------------------------------------------------
    # YOLO TRACKING
    # --------------------------------------------------------

    results = model.track(
        frame,
        persist=True,
        conf=CONFIDENCE,
        classes=list(VEHICLE_CLASSES),
        verbose=False
    )

    result = results[0]

    objects = []

    # --------------------------------------------------------
    # EXTRACT DETECTIONS
    # --------------------------------------------------------

    if result.boxes is not None:

        boxes = result.boxes

        for i in range(len(boxes)):

            xyxy = boxes.xyxy[i].cpu().numpy()

            x1, y1, x2, y2 = map(int, xyxy)

            cls = int(boxes.cls[i].item())

            conf = float(boxes.conf[i].item())

            # Track ID
            if boxes.id is not None:
                track_id = int(boxes.id[i].item())
            else:
                track_id = i

            center = center_of_box(
                (x1, y1, x2, y2)
            )

            objects.append({
                "id": track_id,
                "class": cls,
                "confidence": conf,
                "box": (x1, y1, x2, y2),
                "center": center
            })

    # --------------------------------------------------------
    # CALCULATE OBJECT SPEED
    # --------------------------------------------------------

    for obj in objects:

        track_id = obj["id"]
        center = obj["center"]

        speed = 0.0

        if track_id in previous_positions:
            speed = distance(
                center,
                previous_positions[track_id]
            )

        previous_speeds[track_id] = current_speeds.get(
            track_id,
            speed
        )

        current_speeds[track_id] = speed

        previous_positions[track_id] = center

        obj["speed"] = speed

    # --------------------------------------------------------
    # DETECT POSSIBLE COLLISION
    # --------------------------------------------------------

    collision_detected = False

    collision_objects = set()

    for i in range(len(objects)):

        for j in range(i + 1, len(objects)):

            obj_a = objects[i]
            obj_b = objects[j]

            box_a = obj_a["box"]
            box_b = obj_b["box"]

            center_a = obj_a["center"]
            center_b = obj_b["center"]

            # Distance between objects
            center_distance = distance(
                center_a,
                center_b
            )

            # IoU
            iou = calculate_iou(
                box_a,
                box_b
            )

            # Size-based distance threshold
            area_a = box_area(box_a)
            area_b = box_area(box_b)

            reference_size = math.sqrt(
                max(1, min(area_a, area_b))
            )

            close_objects = (
                center_distance <
                reference_size * 1.5
            )

            # Sudden speed change
            old_speed_a = previous_speeds.get(
                obj_a["id"],
                0
            )

            old_speed_b = previous_speeds.get(
                obj_b["id"],
                0
            )

            speed_change_a = abs(
                obj_a["speed"] - old_speed_a
            )

            speed_change_b = abs(
                obj_b["speed"] - old_speed_b
            )

            sudden_change = (
                speed_change_a >= MIN_SPEED_CHANGE
                or
                speed_change_b >= MIN_SPEED_CHANGE
            )

            # Collision logic
            possible_collision = (
                iou >= COLLISION_IOU
                or close_objects
            )

            if possible_collision and sudden_change:

                collision_detected = True

                collision_objects.add(
                    obj_a["id"]
                )

                collision_objects.add(
                    obj_b["id"]
                )

    # --------------------------------------------------------
    # ACCIDENT STATE
    # --------------------------------------------------------

    if collision_detected:

        accident_active = True
        accident_counter = ACCIDENT_HOLD_FRAMES

        start_incident_recording()

    else:

        if accident_counter > 0:
            accident_counter -= 1
        else:
            accident_active = False

    # --------------------------------------------------------
    # DRAW OBJECTS
    # --------------------------------------------------------

    for obj in objects:

        x1, y1, x2, y2 = obj["box"]

        track_id = obj["id"]

        conf = obj["confidence"]

        is_collision_object = (
            track_id in collision_objects
        )

        if is_collision_object:

            box_color = (0, 0, 255)

        else:

            box_color = (0, 255, 0)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            box_color,
            2
        )

        label = (
            f"ID:{track_id} "
            f"{conf:.2f}"
        )

        draw_label(
            frame,
            label,
            (x1, y1),
            box_color
        )

        # Center point
        cx, cy = map(int, obj["center"])

        cv2.circle(
            frame,
            (cx, cy),
            4,
            box_color,
            -1
        )

    # --------------------------------------------------------
    # ACCIDENT WARNING
    # --------------------------------------------------------

    if accident_active:

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (width, 100),
            (0, 0, 255),
            -1
        )

        frame = cv2.addWeighted(
            overlay,
            0.35,
            frame,
            0.65,
            0
        )

        cv2.putText(
            frame,
            "!!! ACCIDENT DETECTED !!!",
            (30, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            "Check the incident immediately",
            (30, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    else:

        cv2.putText(
            frame,
            "STATUS: NORMAL",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (width - 150, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # WRITE INCIDENT VIDEO
    # --------------------------------------------------------

    if incident_writer is not None:

        incident_writer.write(frame)

        incident_remaining_frames -= 1

        if incident_remaining_frames <= 0:

            finish_incident_recording()

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    cv2.imshow(
        "YOLO Accident Detection",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

finish_incident_recording()

cap.release()

cv2.destroyAllWindows()

print("Program selesai.")