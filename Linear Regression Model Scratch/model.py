import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

X = 2 * np.random.rand(100,1)
y = 8 + -2 * X + np.random.randn(100, 1)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def gradient_descent(X, y, learning_rate = 0.01, epochs = 1000):
    n = len(y)
    w = np.random.randn(1)
    b = np.random.randn(1)
    
    for i in range(epochs):
        y_pred = w * X + b
        dw = -(2/n) * np.sum(X * (y - y_pred))
        db = -(2/n) * np.sum(y - y_pred)
        
        w -= learning_rate * dw
        b -= learning_rate * db
        
    return w, b

w, b = gradient_descent(X, y)

print(f"Optimized Weight: {w}, Optimized Bias: {b}")

y_pred = w * X + b

plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, y_pred, color='red', label="Predicted Line")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
