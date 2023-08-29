import numpy as np
import Models

# Training Set
x = np.array([1, 2, 3])
y = np.array([250, 750, 1125])

# Parameters for gradient descent
alpha = 1.0e-2
num_iterations = 4000
starting_w = 0
starting_b = 0

model = Models.LinearRegression(x, y)
final_w, final_b = model.gradient_descent(starting_w, starting_b, alpha, num_iterations)

print(f"w: {final_w}  b: {final_b}")




