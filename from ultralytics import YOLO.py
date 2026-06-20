from ultralytics import YOLO

model = YOLO(r"C:\Users\LENOVO\Downloads\sackbag_126epoches_04112025.pt")

print(model.names)