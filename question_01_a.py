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
