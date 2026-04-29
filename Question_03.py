import cv2
import numpy as np

# Parameters
N = 6                    # We need at least 4, but 6 is better
p1 = np.zeros((N, 2), dtype=np.float32)
p2 = np.zeros((N, 2), dtype=np.float32)
n = 0

def draw_circle(event, x, y, flags, param):
    global n
    img = param[0]
    pts = param[1]
    
    if event == cv2.EVENT_LBUTTONDOWN and n < N:
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        pts[n] = [x, y]
        n += 1
        print(f"Point {n}: ({x}, {y})")

# Load images
im1 = cv2.imread('c1.jpg')
im2 = cv2.imread('c2.jpg')

if im1 is None or im2 is None:
    print("Error: Could not load images. Check path!")
    exit()

im1_copy = im1.copy()
im2_copy = im2.copy()

# Click on Image 1
cv2.namedWindow('Image 1 - Click 6 points')
param1 = [im1_copy, p1]
cv2.setMouseCallback('Image 1 - Click 6 points', draw_circle, param1)

print("Click 6 corresponding points on Image 1...")
while n < N:
    cv2.imshow('Image 1 - Click 6 points', im1_copy)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cv2.destroyAllWindows()

# Reset for Image 2
n = 0
cv2.namedWindow('Image 2 - Click 6 corresponding points')
param2 = [im2_copy, p2]
cv2.setMouseCallback('Image 2 - Click 6 corresponding points', draw_circle, param2)

print("Now click the EXACT corresponding 6 points on Image 2...")
while n < N:
    cv2.imshow('Image 2 - Click 6 corresponding points', im2_copy)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()

print("\nPoints on Image 1:\n", p1)
print("Points on Image 2:\n", p2)

# Save points (optional)
np.save('points_p1.npy', p1)
np.save('points_p2.npy', p2)