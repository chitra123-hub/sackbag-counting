from ultralytics import YOLO
import cv2

video_path = r"c:\Users\LENOVO\Downloads\vlc-record-2026-06-17-10h21m53s-gulati_ch2_20250922104551_20250922112354.mp4-.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print("Video Width :", width)
print("Video Height:", height)
print("FPS         :", fps)

ret, frame = cap.read()

if not ret:
    print("Could not read first frame")
    cap.release()
    exit()

print("Frame Shape :", frame.shape)

#cv2.imshow("First Frame", frame)

print("Press any key while First Frame window is selected...")

#cv2.waitKey(0)

#cv2.destroyAllWindows()
cap.release()
