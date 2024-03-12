from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np

class DroneDetect:

    def __init__(self, model_path, iou, conf) -> None:
        self.model_path = model_path
        self.iou = iou
        self.conf = conf
        self.classes = {0: 'helicopter', 1: 'UAV', 2: 'airplane', 3: 'birds', 4: 'drone'}

    def __load_model(self):
        model = YOLO(self.model_path)
        return model
    
    def __predict(self, model, image):
        results = model.predict(image, conf = self.conf, iou = self.iou)
        return results
    
    def __draw_boxes(self, image, results):
        for result in results:
            cls = self.classes[int(result.boxes.cls[0])]
            x1, y1, x2, y2 = [int(x) for x in result.boxes.xyxy[0]]
            cv2.rectangle(image, [x1, y1], [x2, y2], color = [0, 0, 0], thickness = 2)
        cv2.imwrite(f'./predictions/{cls}_{np.random.randint(100000)}.png', image)

    def pipeline(self, image_path):
        model = self.__load_model()
        results = self.__predict(model, image_path)
        self.__draw_boxes(cv2.imread(image_path), results)
        return results

if __name__ == '__main__':
    image = 'DC34CEE3-2BF4-4F81-BB01924CDA1B549B_source.jpg'
    drone_detect = DroneDetect(model_path = '/Users/nikolay/Documents/repos/whybee-coding/models/best.pt', iou = 0.6, conf = 0.5)
    results = drone_detect.pipeline(image)
