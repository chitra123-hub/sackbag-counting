from ultralytics import YOLO
import cv2

# Load model
model = YOLO(r"c:\Users\LENOVO\Downloads\anpr_26_05_2026_v1.pt")

# Image path
image_path = r"c:\Users\LENOVO\Downloads\Screenshot 2026-06-01 154320.png"

# Prediction
results = model.predict(
    source=image_path,
    conf=0.4,
    iou=0.45,
    imgsz=640,
    device="cpu"  # Use CPU
)

# Get first result
result = results[0]

# Draw bounding boxes and labels
annotated_image = result.plot(
    line_width=8,
    font_size=2
)

# Save annotated image
output_path = r"D:\code\output.jpg"
cv2.imwrite(output_path, annotated_image)

# Show annotated image
cv2.imshow("Detection Result", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Annotated image saved at: {output_path}")