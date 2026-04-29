import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
im1 = cv2.imread('c1.jpg', cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread('c2.jpg', cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(im1, None)
kp2, des2 = sift.detectAndCompute(im2, None)

# BFMatcher with ratio test
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print(f"Number of good matches: {len(good_matches)}")

# Draw matches
img_matches = cv2.drawMatches(im1, kp1, im2, kp2, good_matches[:50], None, 
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imwrite('sift_matches.jpg', img_matches)

# Compute homography using good matches
if len(good_matches) >= 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    H_auto, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    print("Automatic Homography computed successfully.")
    print("Inliers:", np.sum(mask))
    
    # Warp
    height, width = im2.shape
    im1_warped_auto = cv2.warpPerspective(cv2.imread('c1.jpg'), H_auto, (width, height))
    
    diff_auto = cv2.absdiff(cv2.imread('c2.jpg'), im1_warped_auto)
    
    cv2.imwrite('im1_warped_sift.jpg', im1_warped_auto)
    cv2.imwrite('difference_sift.jpg', diff_auto)
    
    print("Saved: im1_warped_sift.jpg and difference_sift.jpg")
else:
    print("Not enough good matches!")

# Show matches
plt.figure(figsize=(12,6))
plt.imshow(img_matches)
plt.title('SIFT Matches')
plt.axis('off')
plt.show()