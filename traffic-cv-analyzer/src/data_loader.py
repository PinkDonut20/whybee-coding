import cv2

def load_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open {path}")
    return cap

def preprocess_frame(frame, size=(640, 480)):
    return cv2.resize(frame, size)
