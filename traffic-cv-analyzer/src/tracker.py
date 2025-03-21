import numpy as np
from norfair import Detection, Tracker

def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)

class VehicleTracker:
    def __init__(self):
        self.tracker = Tracker(distance_function=euclidean_distance, distance_threshold=30)

    def update(self, detections):
        norfair_detections = [
            Detection(points=np.array([(d["bbox"][0] + d["bbox"][2]) / 2, (d["bbox"][1] + d["bbox"][3]) / 2]),
                      scores=np.array([d["conf"]])) for d in detections
        ]
        return self.tracker.update(norfair_detections)
