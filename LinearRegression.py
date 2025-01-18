import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
x = np.loadtxt('linearX.csv', delimiter=',')
y = np.loadtxt('linearY.csv', delimiter=',')

# Normalize x
x = (x - np.mean(x)) / np.std(x)

# Cost function
def cost_fn(t0, t1, x, y):
    m = len(y)
    preds = t0 + t1 * x
    err = preds - y
    cost = (1 / (2 * m)) * np.sum(err ** 2)
    return cost

# Gradient descent
def grad_desc(x, y, t0, t1, lr, iters):
    m = len(y)
    cost_hist = []

    for _ in range(iters):
        preds = t0 + t1 * x
        err = preds - y
        
        # Gradients
        grad_t0 = (1 / m) * np.sum(err)
        grad_t1 = (1 / m) * np.sum(err * x)
        
        # Update parameters
        t0 -= lr * grad_t0
        t1 -= lr * grad_t1
        
        # Compute cost
        cost = cost_fn(t0, t1, x, y)
        cost_hist.append(cost)
    
    return t0, t1, cost_hist

# Parameters
lr = 0.5  # learning rate
iters = 1000
t0, t1 = 0, 0  # Initial values

# Run gradient descent
t0, t1, cost_hist = grad_desc(x, y, t0, t1, lr, iters)

# Convergence details
print(f"Converged:\nTheta0: {t0}, Theta1: {t1}, Final Cost: {cost_hist[-1]}")

# Plot Cost vs Iterations
plt.plot(range(50), cost_hist[:50])
plt.title("Cost vs Iterations (First 50 Iterations)")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()

# Plot data and regression line
plt.scatter(x, y, label="Data")
plt.plot(x, t0 + t1 * x, color='red', label="Regression Line")
plt.title("Linear Regression Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# Test different learning rates
lrs = [0.005, 0.5, 5]
for rate in lrs:
    _, _, cost_hist = grad_desc(x, y, 0, 0, rate, 50)
    plt.plot(range(50), cost_hist, label=f"lr = {rate}")

plt.title("Cost vs Iterations (Learning Rates)")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.legend()
plt.show()

# Stochastic Gradient Descent
def sgd(x, y, t0, t1, lr, iters):
    m = len(y)
    cost_hist = []

    for _ in range(iters):
        for i in range(m):
            pred = t0 + t1 * x[i]
            err = pred - y[i]

            # Update parameters
            t0 -= lr * err
            t1 -= lr * err * x[i]
        
        cost = cost_fn(t0, t1, x, y)
        cost_hist.append(cost)
    
    return t0, t1, cost_hist

# Mini-Batch Gradient Descent
def mbgd(x, y, t0, t1, lr, iters, batch_size):
    m = len(y)
    cost_hist = []

    for _ in range(iters):
        indices = np.random.permutation(m)
        x_shuff = x[indices]
        y_shuff = y[indices]

        for i in range(0, m, batch_size):
            x_batch = x_shuff[i:i + batch_size]
            y_batch = y_shuff[i:i + batch_size]

            preds = t0 + t1 * x_batch
            err = preds - y_batch

            # Gradients
            grad_t0 = (1 / len(y_batch)) * np.sum(err)
            grad_t1 = (1 / len(y_batch)) * np.sum(err * x_batch)

            # Update parameters
            t0 -= lr * grad_t0
            t1 -= lr * grad_t1
        
        cost = cost_fn(t0, t1, x, y)
        cost_hist.append(cost)
    
    return t0, t1, cost_hist

# Run SGD
t0_sgd, t1_sgd, cost_sgd = sgd(x, y, 0, 0, 0.5, 50)

# Run Mini-Batch GD
t0_mbgd, t1_mbgd, cost_mbgd = mbgd(x, y, 0, 0, 0.5, 50, batch_size=10)

# Compare cost across methods
plt.plot(range(50), cost_hist[:50], label="Batch GD")
plt.plot(range(50), cost_sgd[:50], label="SGD")
plt.plot(range(50), cost_mbgd[:50], label="Mini-Batch GD")
plt.title("Cost vs Iterations (GD Methods)")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.legend()
plt.show()
