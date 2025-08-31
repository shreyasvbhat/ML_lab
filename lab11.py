import numpy as np 
 
# Define the Sigmoid activation function and its derivative (for backpropagation) 
def sigmoid(x): 
    return 1 / (1 + np.exp(-x)) 
 
def sigmoid_derivative(x): 
    return x * (1 - x) 
 
class Perceptron: 
    def __init__(self, input_size): 
        # Initialize the weights with random values and a bias term 
        self.weights = np.random.rand(input_size)  # Random initialization of weights 
        self.bias = np.random.rand(1)  # Random bias initialization 
 
    def forward(self, inputs): 
        # Weighted sum (dot product) + bias 
        total_input = np.dot(inputs, self.weights) + self.bias 
        # Apply the activation function (sigmoid) 
        output = sigmoid(total_input) 
        return output 
 
    def train(self, X, y, epochs=1000, learning_rate=0.1): 
        # Training the perceptron with the perceptron learning rule 
        for epoch in range(epochs): 
            for i in range(X.shape[0]): 
                # Forward pass 
                output = self.forward(X[i]) 
                # Calculate the error (difference between expected and predicted output) 
                error = y[i] - output 
                # Update the weights and bias using the perceptron learning rule 
                self.weights += learning_rate * error * X[i] 
                self.bias += learning_rate * error 
 
# AND and OR dataset 
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input for AND/OR functions 
y_and = np.array([0, 0, 0, 1])  # Expected output for AND function 
y_or = np.array([0, 1, 1, 1])  # Expected output for OR function 
 
# Create perceptron instances for AND and OR 
perceptron_and = Perceptron(input_size=2) 
perceptron_or = Perceptron(input_size=2) 
 
# Train the perceptrons 
perceptron_and.train(X_and, y_and, epochs=1000, learning_rate=0.1) 
perceptron_or.train(X_and, y_or, epochs=1000, learning_rate=0.1) 
 
# Test the perceptrons 
print("AND Function Predictions:") 
for i in range(X_and.shape[0]): 
    print(f"Input: {X_and[i]} - Predicted Output: {round(perceptron_and.forward(X_and[i]))}") 
print("\nOR Function Predictions:") 
for i in range(X_and.shape[0]):
    print(f"Input: {X_and[i]} - Predicted Output: {round(perceptron_or.forward(X_and[i]))}") 