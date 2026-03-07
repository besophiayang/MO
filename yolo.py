import cv2
import time
from picamera2 import Picamera2
from ultralytics import YOLO

picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 640)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

model = YOLO("/home/besophiayang/yolov8n_ncnn_model")

prev_time = time.time()

while True:
    frame = picam2.capture_array()

    results = model.predict(frame, imgsz=320, verbose=False)

    annotated_frame = results[0].plot()

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    text = f"FPS: {fps:.1f}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = annotated_frame.shape[1] - text_size[0] - 10
    text_y = text_size[1] + 10

    cv2.putText(annotated_frame, text, (text_x, text_y),
                font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Camera", annotated_frame)

    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()