from ultralytics import YOLO
import cv2

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch



# model = YOLO('yolov8n-seg.yaml')
model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)
# Use the model
model.train(data="/home/david/AIBirds/src/ab/python_agent/vision/yolo_model/config.yaml", epochs=100, imgsz=640)

# model = YOLO('/home/david/AIBirds/runs/segment/train8/weights/best.pt')
results = model('/home/david/AIBirds/src/ab/python_agent/vision/dataset/images/train/0.jpg', save=True)
