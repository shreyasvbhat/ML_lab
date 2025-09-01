import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)
n = 100
data = pd.DataFrame({
  'X': np.random.normal(0, 1, n),
  'Y': np.random.normal(0, 1, n),
  'Z': np.random.normal(0, 1, n),
  'Category': np.random.choice(['A', 'B', 'C'], n)
})

print(data)

def heatmap():
  correlation = data[['X', 'Y', 'Z']].corr()
  plt.figure(figsize=(5, 4))
  sns.heatmap(correlation, annot=True, cmap='coolwarm')
  plt.title("Heatmap of Correlation Matrix")
  plt.show()

heatmap()


def minimax(depth, node_index, is_maximizing_player, scores, target_depth):
  if depth == target_depth:
    return scores[node_index]
  
  if is_maximizing_player:
    return max(
      minimax(depth + 1, node_index * 2, False, scores, target_depth),
      minimax(depth + 1, node_index * 2 + 1, False, scores, target_depth)
    )
  else:
    return min(
      minimax(depth + 1, node_index * 2, True, scores, target_depth),
      minimax(depth + 1, node_index * 2 + 1, True, scores, target_depth)
    )

print("Enter the depth of the game tree (e.g., 3 for 8 leaf nodes):")
tree_depth = int(input("Depth: "))
num_leaves = 2 ** tree_depth

print(f"Enter {num_leaves} leaf node scores separated by space:")
scores_input = input("Scores: ")
scores = [int(score) for score in scores_input.split(" ")]

if len(scores) != num_leaves:
  print(f"Error: Expected {num_leaves} scores, but go {len(scores)}")
else:
  optimal_value = minimax(0, 0, True, scores, tree_depth)
  print(f"Optimal value using Minimax: {optimal_value}")
