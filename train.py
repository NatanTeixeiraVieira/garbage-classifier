from ultralytics import YOLO

model = YOLO("yolo11n-cls.pt")

model.train(
    data="dataset/train", 
    epochs=100,            
    imgsz=224             
)