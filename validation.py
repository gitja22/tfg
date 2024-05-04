from ultralytics import YOLO

model = YOLO('yolov8n.yaml')
model.train(data='coco128.yaml', epochs=5)
model.val()  # It'll automatically evaluate the data you trained.