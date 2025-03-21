import cv2

def draw_detections(frame, detections):
    for det in detections:
        bbox = det['bbox']
        cv2.rectangle(frame,
                      (int(bbox[0]), int(bbox[1])),
                      (int(bbox[2]), int(bbox[3])),
                      (0, 255, 0), 2)
    return frame

def draw_tracks(frame, tracked_objects):
    for obj in tracked_objects:
        pos = obj.estimate
        cv2.circle(frame, (int(pos[0]), int(pos[1])), 5, (255, 0, 0), -1)
        cv2.putText(frame, f'ID: {obj.id}', (int(pos[0]), int(pos[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return frame
