from ultralytics import YOLO

model = YOLO('./pit_detector.pt')

res = model.predict('./Newport_Whitepit_Lane_pot_hole.JPG')

res.show()