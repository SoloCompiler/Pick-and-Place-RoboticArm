import cv2
import numpy as np

# Replace these with your actual measured pixel coordinates.
image_points = np.array([
    [393,23],    # top-left
    [869 , 7],    # top-right
    [873 , 372],   # bottom-right
    [411 , 380]    # bottom-left
], dtype=np.float32)

# Example workspace: 300 mm wide x 200 mm high
table_points = np.array([
    [0, 0],
    [193, 0],
    [193, 148],
    [0, 198]
], dtype=np.float32)

H = cv2.getPerspectiveTransform(image_points, table_points)

def pixel_to_table(cx, cy):
    point = np.array(
        [[[cx, cy]]], dtype=np.float32
    )

    mapped = cv2.perspectiveTransform(point, H)

    X, Y = mapped[0, 0]
    return float(X), float(Y)

