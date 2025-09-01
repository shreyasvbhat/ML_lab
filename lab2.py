import numpy as np
import pandas as pd
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

def surface_plot():
  fig = plt.figure(figsize=(8, 6))
  ax = fig.add_subplot(111, projection='3d')

  x = np.linspace(-3, 3, 100)
  y = np.linspace(-3, 3, 100)
  X, Y = np.meshgrid(x, y)
  Z = np.sin(np.sqrt(X**2 + Y**2))

  surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
  fig.colorbar(surf)
  ax.set_title("3D Surface Plot")
  plt.show()

surface_plot()

import heapq

class Node:
  def __init__(self, name, heuristic, parent=None):
    self.name = name
    self.heuristic = heuristic
    self.parent = parent

  def __lt__(self, other):
    return self.heuristic < other.heuristic

def best_first_search(graph, start, goal, heuristic_values):
  open_list = []
  closed_list = set()

  heapq.heappush(open_list, Node(start, heuristic_values[start]))

  while open_list:
    current_node = heapq.heappop(open_list)

    if current_node.name == goal:
      path = []
      while current_node:
        path.append(current_node.name)
        current_node = current_node.parent
      return path[::-1]

    if current_node.name in closed_list:
      continue

    closed_list.add(current_node.name)

    for neighbor in graph.get(current_node.name, []):
      if neighbor not in closed_list:
        heapq.heappush(open_list, Node(neighbor, heuristic_values[neighbor], current_node))

  return None

def get_input():
  graph = {}
  heuristic_values = {}

  print("Enter the graph structure:")
  n = int(input("Enter the number of nodes: "))

  for _ in range(n):
    node = input("Enter node name: ")
    neighbors = input(f"Enter neighbors for {node} (comma separated): ").split(",")
    graph[node] = [neighbor.strip() for neighbor in neighbors]

  print("\nEnter heuristic values:")
  for _ in range(n):
    node = input("Enter node name for heuristic: ")
    heuristic = int(input(f"Enter heuristic value for {node}"))
    heuristic_values[node] = heuristic
  
  start = input("\nEnter the start node: ")
  goal = input("Enter the goal node: ")

  return graph, heuristic_values, start, goal

graph, heuristic_values, start, goal = get_input()
path = best_first_search(graph, start, goal, heuristic_values)

if path:
  print(f"\nPath from {start} to {goal}: {path}")
else:
  print(f"\nNo path found from {start} to {goal}")

