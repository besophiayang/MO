import cv2
import time
from ultralytics import YOLO
import RPi.GPIO as GPIO

# Right wheel
in1 = 23
in2 = 22
ena = 24

# Left wheel
in3 = 16
in4 = 26
enb = 17

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

GPIO.setup(in1, GPIO.OUT)
GPIO.setup(in2, GPIO.OUT)
GPIO.setup(in3, GPIO.OUT)
GPIO.setup(in4, GPIO.OUT)
GPIO.setup(ena, GPIO.OUT)
GPIO.setup(enb, GPIO.OUT)

pwm_right = GPIO.PWM(ena, 1000)
pwm_left = GPIO.PWM(enb, 1000)

pwm_right.start(0)
pwm_left.start(0)
MODEL_PATH = "/home/besophiayang/mo_model/best_ncnn_model"   
CAMERA_INDEX = 0

DIRT_CLASS_ID = 0
CONF_THRESH = 0.35

TURN_THRESHOLD_PX = 35
CLOSE_AREA = 18000

BASE_FORWARD_SPEED = 34
BASE_SEARCH_SPEED = 24
MAX_SPEED = 42
MIN_SPEED = 0

STEER_GAIN = 0.10

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

CONTROL_SLEEP = 0.03

# =========================================================
# MOTOR HELPERS
# =========================================================
def clamp(val, low, high):
    return max(low, min(high, val))


def set_right_motor(speed):
    """
    speed in [-100, 100]
    positive = forward
    negative = backward
    """
    speed = clamp(speed, -100, 100)

    if speed > 0:
        GPIO.output(in1, GPIO.HIGH)
        GPIO.output(in2, GPIO.LOW)
        pwm_right.ChangeDutyCycle(abs(speed))
    elif speed < 0:
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.HIGH)
        pwm_right.ChangeDutyCycle(abs(speed))
    else:
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.LOW)
        pwm_right.ChangeDutyCycle(0)


def set_left_motor(speed):
    """
    speed in [-100, 100]
    positive = forward
    negative = backward
    """
    speed = clamp(speed, -100, 100)

    if speed > 0:
        GPIO.output(in3, GPIO.HIGH)
        GPIO.output(in4, GPIO.LOW)
        pwm_left.ChangeDutyCycle(abs(speed))
    elif speed < 0:
        GPIO.output(in3, GPIO.LOW)
        GPIO.output(in4, GPIO.HIGH)
        pwm_left.ChangeDutyCycle(abs(speed))
    else:
        GPIO.output(in3, GPIO.LOW)
        GPIO.output(in4, GPIO.LOW)
        pwm_left.ChangeDutyCycle(0)


def drive(left_speed, right_speed):
    set_left_motor(left_speed)
    set_right_motor(right_speed)


def stop():
    drive(0, 0)


def forward(speed=30):
    drive(speed, speed)


def backward(speed=30):
    drive(-speed, -speed)


def left(turn_speed=25):
    drive(-turn_speed, turn_speed)


def right(turn_speed=25):
    drive(turn_speed, -turn_speed)

# =========================================================
# MODEL
# =========================================================
model = YOLO(MODEL_PATH)

# =========================================================
# DETECTION HELPERS
# =========================================================
def choose_best_dirt_detection(result, frame_w, frame_h):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clses = boxes.cls.cpu().numpy()

    best = None
    best_score = -1

    for box, conf, cls_id in zip(xyxy, confs, clses):
        cls_id = int(cls_id)
        if cls_id != DIRT_CLASS_ID:
            continue
        if conf < CONF_THRESH:
            continue

        x1, y1, x2, y2 = box
        x1 = max(0, min(frame_w - 1, int(x1)))
        y1 = max(0, min(frame_h - 1, int(y1)))
        x2 = max(0, min(frame_w - 1, int(x2)))
        y2 = max(0, min(frame_h - 1, int(y2)))

        area = max(0, (x2 - x1)) * max(0, (y2 - y1))

        score = float(conf) * (area ** 0.5)

        if score > best_score:
            best_score = score
            best = (x1, y1, x2, y2, float(conf), cls_id)

    return best


def draw_debug(frame, best_box, action_text, left_cmd, right_cmd):
    h, w = frame.shape[:2]
    frame_center_x = w // 2

    cv2.line(frame, (frame_center_x, 0), (frame_center_x, h), (255, 255, 0), 2)

    left_thresh = frame_center_x - TURN_THRESHOLD_PX
    right_thresh = frame_center_x + TURN_THRESHOLD_PX
    cv2.line(frame, (left_thresh, 0), (left_thresh, h), (100, 100, 255), 1)
    cv2.line(frame, (right_thresh, 0), (right_thresh, h), (100, 100, 255), 1)

    if best_box is not None:
        x1, y1, x2, y2, conf, cls_id = best_box
        box_center_x = int((x1 + x2) / 2)
        area = (x2 - x1) * (y2 - y1)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (box_center_x, int((y1 + y2) / 2)), 5, (0, 0, 255), -1)

        label = f"dirt conf={conf:.2f} area={area}"
        cv2.putText(frame, label, (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(frame, f"ACTION: {action_text}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.putText(frame, f"L={left_cmd:.1f}  R={right_cmd:.1f}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

    return frame

# =========================================================
# CONTROL LOGIC
# =========================================================
def search_for_dirt():
    """
    Slow in-place search turn.
    """
    left_cmd = BASE_SEARCH_SPEED
    right_cmd = -BASE_SEARCH_SPEED
    drive(left_cmd, right_cmd)
    return "search", left_cmd, right_cmd


def chase_dirt(best_box, frame_w):
    x1, y1, x2, y2, conf, cls_id = best_box

    box_center_x = (x1 + x2) / 2.0
    frame_center_x = frame_w / 2.0
    error = box_center_x - frame_center_x
    area = (x2 - x1) * (y2 - y1)

    if area >= CLOSE_AREA:
        stop()
        return "stop_close", 0, 0

    norm_error = error / (frame_w / 2.0)

    steer = STEER_GAIN * error

    left_cmd = BASE_FORWARD_SPEED + steer
    right_cmd = BASE_FORWARD_SPEED - steer

    left_cmd = clamp(left_cmd, MIN_SPEED, MAX_SPEED)
    right_cmd = clamp(right_cmd, MIN_SPEED, MAX_SPEED)

    # If dirt is far off-center, bias harder into steering
    if error < -TURN_THRESHOLD_PX:
        # target is left -> slow left wheel, speed right wheel
        left_cmd = clamp(BASE_FORWARD_SPEED - abs(steer), MIN_SPEED, MAX_SPEED)
        right_cmd = clamp(BASE_FORWARD_SPEED + abs(steer), MIN_SPEED, MAX_SPEED)
        action = "steer_left"
    elif error > TURN_THRESHOLD_PX:
        # target is right -> speed left wheel, slow right wheel
        left_cmd = clamp(BASE_FORWARD_SPEED + abs(steer), MIN_SPEED, MAX_SPEED)
        right_cmd = clamp(BASE_FORWARD_SPEED - abs(steer), MIN_SPEED, MAX_SPEED)
        action = "steer_right"
    else:
        action = "forward_centered"

    drive(left_cmd, right_cmd)
    return action, left_cmd, right_cmd

# =========================================================
# MAIN LOOP
# =========================================================
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                stop()
                time.sleep(0.05)
                continue

            frame_h, frame_w = frame.shape[:2]

            results = model.predict(
                source=frame,
                conf=CONF_THRESH,
                verbose=False
            )

            result = results[0]
            best_dirt = choose_best_dirt_detection(result, frame_w, frame_h)

            if best_dirt is None:
                action, left_cmd, right_cmd = search_for_dirt()
            else:
                action, left_cmd, right_cmd = chase_dirt(best_dirt, frame_w)

            debug_frame = draw_debug(frame.copy(), best_dirt, action, left_cmd, right_cmd)
            cv2.imshow("M-O Dirt Chase", debug_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            time.sleep(CONTROL_SLEEP)

    except KeyboardInterrupt:
        pass

    finally:
        stop()
        cap.release()
        cv2.destroyAllWindows()
        pwm_right.stop()
        pwm_left.stop()
        GPIO.cleanup()


if __name__ == "__main__":
    main()