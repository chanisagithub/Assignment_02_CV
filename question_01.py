import numpy as np
import matplotlib.pyplot as plt

# Load data
D = np.genfromtxt("lines.csv", delimiter=",", skip_header=1)

# Extract first line's points (columns x1, y1)
x = D[:, 0]   # x1
y = D[:, 3]   # y1

def fit_line_tls(x, y):
    """Fit a line using Total Least Squares. Returns (a, b, c) for ax+by+c=0."""
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Center the data
    A = np.column_stack([x - x_mean, y - y_mean])
    
    # SVD
    _, _, Vt = np.linalg.svd(A)
    
    # Last row of Vt = eigenvector of smallest singular value
    a, b = Vt[-1]
    c = -(a * x_mean + b * y_mean)
    
    return a, b, c

a, b, c = fit_line_tls(x, y)
print(f"Line parameters: a={a:.4f}, b={b:.4f}, c={c:.4f}")
print(f"Slope (if b≠0): {-a/b:.4f}, Intercept: {-c/b:.4f}")

# Plot
plt.figure()
plt.scatter(x, y, label='Data points')
x_range = np.linspace(x.min(), x.max(), 100)
y_range = -(a * x_range + c) / b
plt.plot(x_range, y_range, 'r-', label='TLS fit')
plt.legend(); plt.title("TLS Line Fit (Line 1)"); plt.show()


# --------------Part B-----------------

def ransac_line(x, y, n_iters=500, threshold=0.3, min_inliers=10):
    """Find best line using RANSAC. Returns (a,b,c) and inlier mask."""
    best_inliers = None
    best_count = 0
    n = len(x)
    
    for _ in range(n_iters):
        # Pick 2 random points
        idx = np.random.choice(n, 2, replace=False)
        a, b, c = fit_line_tls(x[idx], y[idx])
        
        # Perpendicular distance of all points to this line
        dist = np.abs(a * x + b * y + c)  # a,b already normalized
        
        inliers = dist < threshold
        count = np.sum(inliers)
        
        if count > best_count:
            best_count = count
            best_inliers = inliers
    
    # Refit using all inliers
    a, b, c = fit_line_tls(x[best_inliers], y[best_inliers])
    return a, b, c, best_inliers


# Flatten all columns as instructed
X_cols = D[:, :3]
Y_cols = D[:, 3:]
X_all = X_cols.flatten()
Y_all = Y_cols.flatten()

lines = []
mask = np.ones(len(X_all), dtype=bool)  # All points available initially
colors = ['red', 'green', 'blue']

plt.figure(figsize=(8,6))
plt.scatter(X_all, Y_all, c='gray', s=10, label='All points')

remaining_x = X_all.copy()
remaining_y = Y_all.copy()
remaining_idx = np.arange(len(X_all))

for i in range(3):
    a, b, c, inliers = ransac_line(remaining_x, remaining_y)
    lines.append((a, b, c))
    print(f"Line {i+1}: a={a:.4f}, b={b:.4f}, c={c:.4f}  "
          f"(slope={-a/b:.4f}, intercept={-c/b:.4f})")
    
    # Plot inliers and line
    plt.scatter(remaining_x[inliers], remaining_y[inliers],
                c=colors[i], s=15, label=f'Line {i+1} inliers')
    x_rng = np.linspace(remaining_x[inliers].min(), remaining_x[inliers].max(), 100)
    plt.plot(x_rng, -(a*x_rng + c)/b, color=colors[i], linewidth=2)
    
    # Mask out consensus set and continue
    remaining_x = remaining_x[~inliers]
    remaining_y = remaining_y[~inliers]

plt.legend()
plt.title("RANSAC: 3 Lines")
plt.show()