import numpy as np

class ViolationChecker:
    def __init__(self, speed_limit=60):
        self.speed_limit = speed_limit

    def check_speed(self, tracked_objects, fps):
        violations = []
        for obj in tracked_objects:
            speed = self.calculate_speed(obj, fps)
            if speed > self.speed_limit:
                violations.append({"id": obj.id, "speed": speed})
        return violations

    def calculate_speed(self, obj, fps):
        positions = obj.estimate
        if len(obj.trajectory) < 2:
            return 0
        p1 = obj.trajectory[-2].estimate
        p2 = obj.trajectory[-1].estimate
        dist_px = np.linalg.norm(np.array(p2) - np.array(p1))
        speed = dist_px * fps  # Примерная скорость
        return speed
