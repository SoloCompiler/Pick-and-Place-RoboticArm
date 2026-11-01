"""
Camera -> table coordinates for pick_and_place.py  (pick_and_place.py imports this file).

    python vision_bridge.py calibrate   # once: click known table points, type their real x y  (redo if the camera moves)
    python vision_bridge.py test        # live view: every detected object with its table position in cm

How it works
  1. detect_blobs():  colourful blobs on a plain table  ->  (u, v, angle) in IMAGE pixels
  2. a 3x3 homography H (from `calibrate`) turns pixel (u, v) into table (x, y) in metres, in the ROBOT frame:
        origin = on the table under the base axis, X forward, Y to the left.
  3. pick_and_place.py's CameraDetector does steps 1-2 for every detection and hands (x, y, yaw) to the planner.

Camera tips: mount it looking STRAIGHT DOWN, fixed, with steady light and a plain, non-saturated table
(white / grey / black).  The homography is exact only for the plane you calibrated on: put the reference
points at the height of the TOP of the objects (e.g. stand each marker on a block as tall as the cube).
"""
import json
import math
import os
import sys

import numpy as np
import cv2

# ---- settings -----------------------------------------------------------------------------------
CAM_INDEX = 0                      # 0 = first camera; try 1, 2 if you have several
CAM_SIZE = (1280, 720)
CALIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_calibration.json")

MIN_SAT, MIN_VAL = 90, 70          # a pixel is "object" if colourful (saturation) and not too dark
MIN_AREA_PX, MAX_AREA_PX = 250, 30000
# To look only for certain colours, list HSV ranges (OpenCV: H 0-179, S 0-255, V 0-255), e.g.
#   COLOR_RANGES = [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (179, 255, 255))]   # red
COLOR_RANGES = None


# ---- detection ----------------------------------------------------------------------------------
def detect_blobs(frame_bgr):
    """Return [(u, v, angle_rad), ...]: centre of every object blob and its edge direction (image axes)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    if COLOR_RANGES:
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in COLOR_RANGES:
            mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    else:
        mask = cv2.inRange(hsv, np.array((0, MIN_SAT, MIN_VAL), np.uint8), np.array((179, 255, 255), np.uint8))
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    out = []
    for c in contours:
        if not (MIN_AREA_PX <= cv2.contourArea(c) <= MAX_AREA_PX):
            continue
        (cx, cy), (w, h), ang = cv2.minAreaRect(c)
        if w < h:
            ang += 90.0
        out.append((float(cx), float(cy), math.radians(ang)))     # cubes repeat every 90 deg - the planner knows
    return out


class Camera:
    def __init__(self, index=CAM_INDEX):
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            raise SystemExit(f"Cannot open camera {index}. Close other programs using it or change CAM_INDEX.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_SIZE[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_SIZE[1])

    def frame(self, flush=5):
        for _ in range(flush):                    # throw away old buffered frames: we want what is there NOW
            self.cap.grab()
        ok, f = self.cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        return f


# ---- calibration --------------------------------------------------------------------------------
def fit_homography(px, table_m):
    """Least-squares homography from >= 4 pixel points to table points (metres). Returns (H, errors_mm)."""
    px, table_m = np.float32(px), np.float32(table_m)
    H, _ = cv2.findHomography(px, table_m, 0)
    if H is None:
        raise ValueError("could not fit a homography - are the points in a line?")
    pts = np.hstack([px, np.ones((len(px), 1), np.float32)]) @ H.T
    proj = pts[:, :2] / pts[:, 2:3]
    return H, 1000 * np.linalg.norm(proj - table_m, axis=1)


def load_homography():
    if not os.path.exists(CALIB_FILE):
        raise SystemExit("No camera calibration yet.  Run:  python vision_bridge.py calibrate")
    with open(CALIB_FILE) as f:
        return np.array(json.load(f)["H"], float)


def calibrate():
    cam = Camera()
    img = cam.frame()
    clicks = []
    win = "CALIBRATE: click each reference point, Enter = done, z = undo, Esc = cancel"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print("Click at least 4 points whose position on the table you know (corners of a printed grid, marks measured\n"
          "from the base axis with a ruler). Spread them over the whole working area; 6-8 is better than 4.")
    while True:
        view = img.copy()
        for i, (x, y) in enumerate(clicks, 1):
            cv2.circle(view, (x, y), 6, (0, 0, 255), 2)
            cv2.putText(view, str(i), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow(win, view)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            raise SystemExit("cancelled")
        if k == ord("z") and clicks:
            clicks.pop()
        if k in (13, 10) and len(clicks) >= 4:
            break
    cv2.destroyAllWindows()

    table = []
    print("\nNow type the real position of each point in CM, robot frame (X forward from the base axis, Y to the left).")
    for i in range(len(clicks)):
        while True:
            try:
                x, y = [float(t) for t in input(f"  point {i+1}  x y (cm): ").replace(",", " ").split()]
                table.append((x / 100, y / 100))
                break
            except ValueError:
                print("   two numbers please, e.g.  20 -5")
    H, err = fit_homography(clicks, table)
    print("\nReprojection error per point (mm):", " ".join(f"{e:.1f}" for e in err))
    if err.max() > 8:
        print("WARNING: error above 8 mm - a point was clicked or typed wrongly. Run calibrate again.")
    with open(CALIB_FILE, "w") as f:
        json.dump({"H": H.tolist(), "pixels": clicks, "table_m": table}, f, indent=1)
    print("Saved", CALIB_FILE)


# ---- the hook pick_and_place.py uses ----------------------------------------------------------------
def make_detector():
    """Called by pick_and_place.py in hardware mode."""
    from pick_and_place import CameraDetector
    H = load_homography()
    cam = Camera()
    return CameraDetector(lambda: detect_blobs(cam.frame()), H)


def test():
    from pick_and_place import CameraDetector
    H = load_homography()
    cam = Camera()
    det = CameraDetector(lambda: [], H)
    print("Live view. Press q to quit.")
    while True:
        frame = cam.frame(flush=1)
        for u, v, ang in detect_blobs(frame):
            x, y = det._to_table(u, v)
            cv2.circle(frame, (int(u), int(v)), 5, (0, 255, 0), -1)
            cv2.line(frame, (int(u), int(v)), (int(u + 30 * math.cos(ang)), int(v + 30 * math.sin(ang))), (0, 255, 0), 2)
            cv2.putText(frame, f"{100*x:.1f},{100*y:.1f} cm", (int(u) + 8, int(v) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("vision test (q = quit)", frame)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cmd = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    if cmd == "calibrate":
        calibrate()
    elif cmd == "test":
        test()
    else:
        print(__doc__)
