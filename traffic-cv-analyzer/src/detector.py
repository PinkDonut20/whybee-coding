from ultralytics import YOLO

class VehicleDetector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        results = self.model(frame)[0]
        detections = []
        for box in results.boxes:
            if box.conf > self.conf_threshold:
                detections.append({
                    "bbox": box.xyxy.tolist()[0],
                    "conf": float(box.conf),
                    "cls": int(box.cls)
                })
        return detections
