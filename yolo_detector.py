from ultralytics import YOLO
import os

class YoloDetector:

    def __init__(self):
        pass

    def train(self, path_to_model, data):
        model = YOLO(path_to_model)
        model.train(data = data, epochs = 10)

    def val(self, path_to_model):
        model = YOLO(path_to_model)
        model.val(plots=True)


if __name__ == '__main__':
    detector = YoloDetector()
    detector.train(path_to_model = 'yolov8n.pt', data = './configs/soccer_detection.yaml')
