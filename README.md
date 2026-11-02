# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 1.Import necessary libearies(numpy, matplotlib)
2. Generate synthetic data with a linear relationship and noise
3.  Add bias term to the input features 3. Initialize model parameters randomly
4.   Define gradient decent function to update paramaters iteratively
5.   Train the model using the gradient descent function
6.   Print learned parameters(intercept and slope)
7.   Visualize the results with a scatter plot and regression line
   

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: LITYA M
RegisterNumber: 212225230152 
*/

import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add bias term
X_b = np.c_[np.ones((100, 1)), X]

# Gradient Descent Function
def gradient_descent(X, y, learning_rate=0.1, n_iterations=1000):

    m = len(y)

    # Initialize weights randomly
    theta = np.random.randn(2, 1)

    for iteration in range(n_iterations):

        gradients = 2/m * X.T.dot(X.dot(theta) - y)

        theta = theta - learning_rate * gradients

    return theta


# Train the model
theta_best = gradient_descent(X_b, y)

# Output model parameters
print(f"Learned parameters: intercept = {theta_best[0][0]:.4f}, slope = {theta_best[1][0]:.4f}")

# Plotting
plt.scatter(X, y, color='blue', label='Data')

X_new = np.array([[0], [2]])

X_new_b = np.c_[np.ones((2, 1)), X_new]

y_predict = X_new_b.dot(theta_best)

plt.plot(X_new, y_predict, color='red', label='Prediction')

plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression using Gradient Descent")

plt.legend()
plt.grid(True)
plt.show()

```

## Output:
![linear regression using gradient descent](sam.png)
<img width="965" height="692" alt="image" src="https://github.com/user-attachments/assets/08165f90-9564-41a7-8969-737877315ef7" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
