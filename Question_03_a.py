import cv2
import numpy as np
import matplotlib.pyplot as plt

# ================== Your clicked points ==================
p1 = np.array([
    [310, 2158],
    [1334, 2387],
    [983, 1897],
    [1434, 446],
    [1901, 377],
    [1856, 812]
], dtype=np.float32)

p2 = np.array([
    [136, 1780],
    [1062, 2268],
    [853, 1705],
    [1673, 425],
    [2143, 479],
    [1984, 888]
], dtype=np.float32)

# Load images
im1 = cv2.imread('c1.jpg')
im2 = cv2.imread('c2.jpg')

if im1 is None or im2 is None:
    print("Error loading images! Check the path.")
    exit()

# Compute Homography (from im1 -> im2)
H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)

print("Homography Matrix:")
print(np.round(H, decimals=6))
print("\nInliers:", status.ravel())

# Warp im1 to im2's perspective
height, width = im2.shape[:2]
im1_warped = cv2.warpPerspective(im1, H, (width, height))

# Create difference image (absolute difference)
diff = cv2.absdiff(im2, im1_warped)

# Save results
cv2.imwrite('im1_warped_manual.jpg', im1_warped)
cv2.imwrite('difference_manual.jpg', diff)

print("\nFiles saved: im1_warped_manual.jpg and difference_manual.jpg")

# Display results
plt.figure(figsize=(15, 8))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB))
plt.title('Original c1.jpg')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(im1_warped, cv2.COLOR_BGR2RGB))
plt.title('Warped c1 → c2 perspective')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB))
plt.title('Difference Image (Manual)')
plt.axis('off')

plt.tight_layout()
plt.show()