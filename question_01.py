import numpy as np
import matplotlib.pyplot as plt

D = np.genfromtxt("lines.csv", delimiter=",", skip_header=1)
X1 = D[:, 0]   # x1
Y1 = D[:, 3]   # y1

def total_least_squares_line(x, y):
    # Center the data
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_c = x - x_mean
    y_c = y - y_mean
    
    # Form the matrix
    A = np.column_stack((x_c, y_c))
    
    # SVD
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Normal vector is the last right singular vector
    a, b = Vt[-1]
    
    # Line equation: a(x - x_mean) + b(y - y_mean) = 0
    c = -a * x_mean - b * y_mean
    
    # Normalize so that a^2 + b^2 = 1
    norm = np.sqrt(a**2 + b**2)
    a, b, c = a/norm, b/norm, c/norm
    
    return a, b, c 

# Usage
a, b, c = total_least_squares_line(X1, Y1)
print(f"Line 1 parameters (a, b, c): {a:.6f}, {b:.6f}, {c:.6f}")


D = np.genfromtxt("lines.csv", delimiter=",", skip_header=1)
X_cols = D[:, :3]
Y_cols = D[:, 3:]
X_all = X_cols.flatten()
Y_all = Y_cols.flatten()

points = np.column_stack((X_all, Y_all))   # shape (N, 2)

from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression

def ransac_line_fit(points, residual_threshold=0.4, max_trials=1000):
    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    
    ransac = RANSACRegressor(
        estimator=LinearRegression(),
        min_samples=2,
        residual_threshold=residual_threshold,
        max_trials=max_trials,
        random_state=42
    )
    ransac.fit(X, y)
    
    slope = ransac.estimator_.coef_[0]
    intercept = ransac.estimator_.intercept_
    inlier_mask = ransac.inlier_mask_
    
    return slope, intercept, inlier_mask

remaining_mask = np.ones(len(points), dtype=bool)
lines = []
inlier_groups = []

for i in range(3):
    current_points = points[remaining_mask]
    slope, intercept, inlier_mask = ransac_line_fit(current_points)
    
    # Store the line
    lines.append((slope, intercept))
    
    # Update remaining points
    current_inliers = np.where(remaining_mask)[0][inlier_mask]
    inlier_groups.append(current_inliers)
    remaining_mask[current_inliers] = False
    
    print(f"Line {i+1}: y = {slope:.4f}x + {intercept:.4f}")

colors = ["tab:red", "tab:green", "tab:blue"]
x_line = np.linspace(points[:, 0].min(), points[:, 0].max(), 200)

plt.figure(figsize=(8, 6))
plt.scatter(points[:, 0], points[:, 1], s=18, c="lightgray", label="All points")

for i, ((slope, intercept), inlier_idx) in enumerate(zip(lines, inlier_groups)):
    color = colors[i % len(colors)]
    inlier_pts = points[inlier_idx]
    y_line = slope * x_line + intercept
    plt.scatter(
        inlier_pts[:, 0],
        inlier_pts[:, 1],
        s=22,
        c=color,
        label=f"Line {i+1} inliers",
    )
    plt.plot(x_line, y_line, color=color, linewidth=2.2, label=f"Fitted line {i+1}")

plt.xlabel("x")
plt.ylabel("y")
plt.title("RANSAC inliers and fitted lines")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()
