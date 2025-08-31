# Visualize the n-dimensional data using Scatter plots.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)  # Ensures reproducibility
n = 100
data = pd.DataFrame({
  'X': np.random.normal(0, 1, n),
  'Y': np.random.normal(0, 1, n),
  'Z': np.random.normal(0, 1, n),
  'Category': np.random.choice(['A', 'B', 'C'], n)
})

print(data)

def scatter_plot():
  plt.figure(figsize=(6, 4))
  sns.scatterplot(data=data, x='X', y='Y', hue='Category')
  plt.title("2D Scatter Plot")
  plt.show()

scatter_plot()

# --------------------------------------------------------------------------------

# Write a program to implement Hill Climbing Algorithm.
import random

def objective_function(x):
  return -x ** 2 + 5

def hill_climbing(start_x, step_size, max_iterations):
  current_x = start_x
  current_score = objective_function(current_x)

  for i in range(max_iterations):
    new_x = current_x + random.uniform(-step_size, step_size)
    new_score = objective_function(new_x)

    print(f"Iteration {i + 1}: x = {current_x:.4f}, f(x) = {current_score:.4f}")

    if new_score > current_score:
      current_x = new_x
      current_score = new_score
    else:
      pass
  
  print("\nFinal Solution:")
  print(f"x = {current_x:.4f}, f(x) = {current_score:.4f}")
  return current_x, current_score

best_x, best_score = hill_climbing(start_x=0.1, step_size=0.05, max_iterations=5)



# import numpy as np
# import matplotlib.pyplot as plt
# import random

# # -------- Scatter Plot --------
# def scatter_demo():
#     X = np.random.rand(200, 4)  # 200 samples, 4 features
#     plt.scatter(X[:, 0], X[:, 1], c='blue', alpha=0.6)
#     plt.title("Scatter Plot (first 2 features)")
#     plt.xlabel("Feature 1")
#     plt.ylabel("Feature 2")
#     plt.show()

# # -------- Hill Climbing --------
# def objective(x):
#     return -(x**2) + 5   # simple parabola (max at x=0)

# def hill_climb(start, step=0.1, max_iter=100):
#     current = start
#     current_val = objective(current)
#     for _ in range(max_iter):
#         neighbors = [current - step, current + step]
#         best = max(neighbors, key=objective)
#         if objective(best) > current_val:
#             current, current_val = best, objective(best)
#         else:
#             break
#     return current, current_val

# # Run
# scatter_demo()
# best_x, best_val = hill_climb(random.uniform(-5, 5))
# print("Best x:", best_x, "Value:", best_val)
