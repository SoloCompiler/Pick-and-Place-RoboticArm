# to find the coordinates of the four corners of the workspace in the camera frame
import cv2

cap = cv2.VideoCapture(1)

points = []

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point {len(points)}: ({x}, {y})")

        if len(points) > 4:
            points.pop(0)

cv2.namedWindow("Camera")
cv2.setMouseCallback("Camera", on_mouse)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    for i, point in enumerate(points):
        cv2.circle(frame, point, 6, (0, 0, 255), -1)
        cv2.putText(
            frame, str(i + 1),
            (point[0] + 8, point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2
        )

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
