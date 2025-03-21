from fastapi import FastAPI, UploadFile
from src.data_loader import load_video, preprocess_frame
from src.detector import VehicleDetector
from src.tracker import VehicleTracker
from src.visualizer import draw_detections, draw_tracks
import cv2
import tempfile

app = FastAPI()

@app.post("/analyze/")
async def analyze(file: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(await file.read())
        video_path = temp.name

    cap = load_video(video_path)
    detector = VehicleDetector()
    tracker = VehicleTracker()

    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = preprocess_frame(frame)
        detections = detector.detect(frame)
        tracked_objects = tracker.update(detections)

        frame = draw_detections(frame, detections)
        frame = draw_tracks(frame, tracked_objects)

        frame_count += 1
        if frame_count >= 50:  # Обработаем только 50 кадров для примера
            break

    cap.release()

    return {"message": "Video analyzed", "frames_processed": frame_count}
