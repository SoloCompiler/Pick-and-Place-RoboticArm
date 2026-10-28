
import cv2
import numpy as np
from coordinates import pixel_to_table

cap = cv2.VideoCapture(1)

colors = {
    "green": (
        np.array([35, 70, 50]),
        np.array([85, 255, 255])
    ),
    "blue": (
        np.array([90, 70, 50]),
        np.array([130, 255, 255])
    )
}

# Red has two HSV ranges
red_ranges = [
    (np.array([0, 70, 50]), np.array([10, 255, 255])),
    (np.array([170, 70, 50]), np.array([179, 255, 255]))
]

kernel = np.ones((5, 5), np.uint8)
MIN_AREA = 500

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    masks = {
        name: cv2.inRange(hsv, low, high)
        for name, (low, high) in colors.items()
    }

    masks["red"] = cv2.bitwise_or(
        cv2.inRange(hsv, *red_ranges[0]),
        cv2.inRange(hsv, *red_ranges[1])
    )

    for color, mask in masks.items():
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, kernel
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, kernel
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            if cv2.contourArea(contour) < MIN_AREA:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w // 2
            cy = y + h // 2

            X, Y = pixel_to_table(cx, cy)

            cv2.putText(
            frame,
            f"Table: X={X:.1f} Y={Y:.1f} mm",
            (x, y + h + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

            print(
                f"{color}: pixel=({cx}, {cy}), "
                f"table=({X:.1f}, {Y:.1f}) mm"
            )

            cv2.rectangle(
                frame, (x, y), (x + w, y + h),
                (0, 255, 0), 2
            )

            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            cv2.putText(
                frame, f"{color}: ({cx}, {cy})",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2
            )

    cv2.imshow("Multi-color Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
