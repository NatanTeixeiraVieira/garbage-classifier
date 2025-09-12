from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")

model.train(
    data="dataset/train", 
    epochs=15,            
    imgsz=224             
)