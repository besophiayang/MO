import time
import random
from dataclasses import dataclass

import board
import digitalio
import busio
from PIL import Image, ImageDraw
from adafruit_rgb_display import ili9341


PANEL_W = 240
PANEL_H = 320

CS_PIN = board.D5
DC_PIN = board.D25
RST_PIN = board.D27
BL_PIN = board.D18

DISPLAY_ROTATION = 0

LANDSCAPE_VIEW = "CCW"

BLACK = (0, 0, 0)
YELLOW = (255, 210, 40)
YELLOW_DIM = (180, 140, 20)


LAND_W = 320
LAND_H = 240

VISIBLE_HEIGHT = int(LAND_H * 0.40)   
VISIBLE_Y_START = LAND_H - VISIBLE_HEIGHT  


@dataclass
class FaceState:
    target_x: float = 0.5   
    blink: float = 0.0      


class MOEyesDisplay:
    def __init__(self):
        spi = busio.SPI(clock=board.SCK, MOSI=board.MOSI)

        dc = digitalio.DigitalInOut(DC_PIN)
        cs = digitalio.DigitalInOut(CS_PIN)
        rst = digitalio.DigitalInOut(RST_PIN)

        self.display = ili9341.ILI9341(
            spi,
            rotation=DISPLAY_ROTATION,
            cs=cs,
            dc=dc,
            rst=rst,
            baudrate=32000000,
            width=PANEL_W,
            height=PANEL_H,
        )

        try:
            self.bl = digitalio.DigitalInOut(board.D18)
            self.bl.direction = digitalio.Direction.OUTPUT
            self.bl.value = True
        except Exception:
            self.bl = None

        self.state = FaceState()
        self.last_blink_time = time.monotonic()
        self.next_blink_delay = random.uniform(2.0, 4.0)
        self.blink_duration = 0.18

    def update_from_vision(self, dirt_found=False, dirt_center_x=0.5):
        if dirt_found:
            self.state.target_x = max(0.0, min(1.0, dirt_center_x))
        else:
            self.state.target_x = 0.5

    def _update_blink(self):
        now = time.monotonic()
        elapsed = now - self.last_blink_time

        if elapsed >= self.next_blink_delay:
            blink_elapsed = elapsed - self.next_blink_delay

            if blink_elapsed <= self.blink_duration:
                half = self.blink_duration / 2
                if blink_elapsed < half:
                    self.state.blink = blink_elapsed / half
                else:
                    self.state.blink = 1.0 - ((blink_elapsed - half) / half)
            else:
                self.state.blink = 0.0
                self.last_blink_time = now
                self.next_blink_delay = random.uniform(2.0, 4.0)
        else:
            self.state.blink = 0.0

    def _land_to_panel(self, u, v):
        if LANDSCAPE_VIEW == "CCW":
            x = PANEL_W - 1 - v
            y = u
        else:
            x = v
            y = PANEL_H - 1 - u

        return x, y

    def _rect_landscape(self, draw, u0, v0, u1, v1, fill):
        pts = [
            self._land_to_panel(u0, v0),
            self._land_to_panel(u1, v0),
            self._land_to_panel(u1, v1),
            self._land_to_panel(u0, v1),
        ]
        draw.polygon(pts, fill=fill)

    def render(self):
        self._update_blink()

        image = Image.new("RGB", (PANEL_W, PANEL_H), BLACK)
        draw = ImageDraw.Draw(image)

        region_y0 = VISIBLE_Y_START
        region_h = VISIBLE_HEIGHT

        eye_w = 100
        eye_h_open = 45
        eye_h_closed = 4
        gap = 25

        blink = self.state.blink
        eye_h = int(eye_h_open * (1.0 - blink) + eye_h_closed * blink)

        total_w = eye_w * 2 + gap
        start_x = (LAND_W - total_w) // 2

        y = region_y0 + (region_h - eye_h_open) // 2 + 6

        look = (self.state.target_x - 0.5) * 2.0
        max_shift = 14
        shift_x = int(look * max_shift)

        left_x0 = start_x + shift_x
        left_y0 = y + (eye_h_open - eye_h) // 2
        left_x1 = left_x0 + eye_w
        left_y1 = left_y0 + eye_h

        right_x0 = start_x + eye_w + gap + shift_x
        right_y0 = y + (eye_h_open - eye_h) // 2
        right_x1 = right_x0 + eye_w
        right_y1 = right_y0 + eye_h

        self._rect_landscape(draw, left_x0, left_y0, left_x1, left_y1, YELLOW)
        self._rect_landscape(draw, right_x0, right_y0, right_x1, right_y1, YELLOW)

        if eye_h > 6:
            self._rect_landscape(draw, left_x0, left_y1 - 2, left_x1, left_y1, YELLOW_DIM)
            self._rect_landscape(draw, right_x0, right_y1 - 2, right_x1, right_y1, YELLOW_DIM)

        self.display.image(image)

    def demo_loop(self):
        demo_positions = [0.5, 0.2, 0.8, 0.5]
        i = 0
        last_switch = time.monotonic()

        while True:
            if time.monotonic() - last_switch > 2.0:
                i = (i + 1) % len(demo_positions)
                last_switch = time.monotonic()

            self.update_from_vision(dirt_found=True, dirt_center_x=demo_positions[i])
            self.render()
            time.sleep(0.03)


if __name__ == "__main__":
    face = MOEyesDisplay()
    face.demo_loop()