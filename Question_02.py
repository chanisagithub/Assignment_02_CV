import cv2
import numpy as np

im = cv2.imread("earrings.jpg")
if im is None:
    raise FileNotFoundError("Could not load 'earrings.jpg'. Check the file path.")

window_name = "Earrings - Click two points"
display = im.copy()
points = []


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
        points.append((x, y))
        cv2.circle(display, (x, y), 4, (0, 255, 0), -1)

        if len(points) == 2:
            cv2.line(display, points[0], points[1], (255, 0, 0), 2)
            p1, p2 = np.array(points[0]), np.array(points[1])
            dist_px = np.linalg.norm(p2 - p1)
            print(f"Distance: {dist_px:.2f} pixels")


cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, on_mouse)

while True:
    cv2.imshow(window_name, display)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to exit
        break

cv2.destroyAllWindows()
