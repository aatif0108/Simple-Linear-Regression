# Simple-Linear-Regression 


## Description
This project implements **Simple Linear Regression** using **Gradient Descent** to fit a straight line to a given dataset. It explores various optimization methods such as Batch, Stochastic, and Mini-Batch Gradient Descent. The project includes data preprocessing, detailed analysis of learning rates, and visualizations to evaluate convergence and model performance.

## Key Features
- **Normalization**: The predictor values (`x`) are normalized for faster and more stable convergence.
- **Optimization Methods**:
  - Batch Gradient Descent: Stable but slower for large datasets.
  - Stochastic Gradient Descent: Faster but with fluctuating updates.
  - Mini-Batch Gradient Descent: Balances speed and stability.
- **Learning Rate Analysis**: The model's performance is evaluated for different learning rates (e.g., 0.005, 0.5, and 5) to observe their impact on convergence.
- **Visualizations**:
  - Cost vs. Iterations for different learning rates.
  - Regression line fit to the dataset.
  - Comparison of Gradient Descent methods.

## Results
- The model successfully fits a regression line to the dataset.
- Optimal Learning Rate: **0.5**
- Final Parameters after Convergence:
  - **Theta0** (Intercept): `0.997`
  - **Theta1** (Slope): `0.0013`
  - **Final Cost**: `1.19 × 10⁻⁶`
- Mini-Batch Gradient Descent achieves a balance between speed and stability, making it efficient for larger datasets.

