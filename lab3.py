# Visualize the n-dimensional data using contour plots.
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

def contour_plot():
  x = np.linspace(-3, 3, 100)
  y = np.linspace(-3, 3, 100)
  X, Y = np.meshgrid(x, y)
  Z = np.sin(X**2 + Y**2)

  plt.figure(figsize=(6, 5))
  cp = plt.contourf(X, Y, Z, cmap='viridis')
  plt.colorbar(cp)
  plt.title("Contour Plot of sin(X² + Y²)")
  plt.xlabel("X")
  plt.ylabel("Y")
  plt.show()

contour_plot()

# Write a program to implement the A* algorithm
import heapq

class Node:
  def __init__(self, name, g_cost, h_cost, parent=None):
    self.name = name
    self.g_cost = g_cost
    self.h_cost = h_cost
    self.f_cost = g_cost + h_cost
    self.parent = parent

  def __lt__(self, other):
    return self.f_cost < other.f_cost

def a_star_search(graph, costs, start, goal, heuristic_values):
  open_list = []
  closed_list = set()

  heapq.heappush(open_list, Node(start, 0, heuristic_values[start]))

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
      if neighbor in closed_list:
        continue
      g_cost = current_node.g_cost + costs[(current_node.name, neighbor)]
      h_cost = heuristic_values[neighbor]
      heapq.heappush(open_list, Node(neighbor, g_cost, h_cost, current_node))

  return None

def get_input():
  graph = {}
  heuristic_values = {}
  costs = {}

  print("Enter the graph structure:")
  n = int(input("Enter the number of nodes: "))

  for _ in range(n):
    node = input("Enter node name: ")
    neighbors = input(f"Enter neighbors for {node} (comma separated): ").split(",")
    graph[node] = [neighbor.strip() for neighbor in neighbors]

  print("\nEnter costs between nodes:")
  for node in graph:
    for neighbor in graph[node]:
      cost = int(input(f"Enter cost from {node} to {neighbor}: "))
      costs[(node, neighbor)] = cost

  print("\nEnter heuristic values:")
  for _ in range(n):
    node = input("Enter node name for heuristic: ")
    heuristic = int(input(f"Enter heuristic value for {node}: "))
    heuristic_values[node] = heuristic

  start = input("\nEnter the start node: ")
  goal = input("Enter the goal node: ")

  return graph, costs, heuristic_values, start, goal

graph, costs, heuristic_values, start, goal = get_input()
path = a_star_search(graph, costs, start, goal, heuristic_values)

if path:
  print(f"\nPath from {start} to {goal}: {path}")
else:
  print(f"\nNo path found from {start} to {goal}")

