from ultralytics import YOLO
import cv2

# Load model
model = YOLO(
    r"C:\Users\LENOVO\Downloads\sackbag_126epoches_04112025.pt"
)

# Open video
cap = cv2.VideoCapture(
    r"c:\Users\LENOVO\Downloads\vlc-record-2026-06-17-10h21m53s-gulati_ch2_20250922104551_20250922112354.mp4-.mp4"
)

if not cap.isOpened():
    print("Error opening video")
    exit()

# Video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print("Width :", width)
print("Height:", height)
print("FPS   :", fps)

# Save output video
out = cv2.VideoWriter(
    "output_counted.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

# Counting line position
line_y = 1300

counted_ids = set()
total_count = 0

# Display window
cv2.namedWindow("Counting", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Counting", 1366, 768)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml"
    )

    # Horizontal counting line
    cv2.line(
        frame,
        (0, line_y),
        (width, line_y),
        (0, 255, 0),
        5
    )

    if results and results[0].boxes is not None:

        for box in results[0].boxes:

            cls = int(box.cls[0])

            if model.names[cls] != "sackbags":
                continue

            if box.id is None:
                continue

            track_id = int(box.id[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                3
            )

            # Center point
            cv2.circle(
                frame,
                (cx, cy),
                6,
                (0, 0, 255),
                -1
            )

            # Counting
            if cy > line_y and track_id not in counted_ids:
                counted_ids.add(track_id)
                total_count += 1

    # Count display
    cv2.putText(
        frame,
        f"COUNT: {total_count}",
        (50, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        3,
        (0, 0, 255),
        5
    )

    # Save full resolution
    out.write(frame)

    # Show resized window
    display_frame = cv2.resize(frame, (1366, 768))

    cv2.imshow("Counting", display_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Final Count =", total_count)