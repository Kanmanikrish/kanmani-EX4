import numpy as np

class CNN:
    """
    Simple convolutional neural network with one convolutional layer and one fully-connected layer.
    """
    def __init__(self, input_shape, num_classes, kernel_size=(3, 3), num_filters=8):
        """
        Initializes the CNN with input shape, number of classes, kernel size, and number of filters.

        Args:
          input_shape: A tuple representing the shape of the input image (height, width, channels).
          num_classes: The number of output classes (number of categories to classify).
          kernel_size: A tuple representing the size of the convolutional kernel.
          num_filters: The number of filters in the convolutional layer.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.num_filters = num_filters

        # Initialize weights and biases for convolutional layer and fully-connected layer
        self.conv_weights = np.random.randn(self.kernel_size[0], self.kernel_size[1], input_shape[2], self.num_filters) / np.sqrt(input_shape[2])
        self.conv_bias = np.zeros(shape=(self.num_filters,))
        self.fc_weights = np.random.randn(self.num_filters * input_shape[0] * input_shape[1], self.num_classes) / np.sqrt(self.num_filters * input_shape[0] * input_shape[1])
        self.fc_bias = np.zeros(shape=(self.num_classes,))

    def sigmoid(self, x):
        """
        Sigmoid activation function.
        """
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """
        Softmax activation function.
        """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def forward(self, X):
        """
        Forward pass through the CNN.

        Args:
          X: A numpy array representing the input image(s).

        Returns:
          A numpy array representing the output probabilities for each class.
        """
        # Convolutional layer
        conv_out = np.zeros(shape=(X.shape[0], X.shape[1] - self.kernel_size[0] + 1, X.shape[2] - self.kernel_size[1] + 1, self.num_filters))
        for i in range(X.shape[0]):
            for f in range(self.num_filters):
                for c in range(X.shape[2]):
                    for j in range(self.kernel_size[0]):
                        for k in range(self.kernel_size[1]):
                            # Apply filter to input image
                            conv_out[i, j, k, f] += X[i, j + self.kernel_size[0] - 1 - k, c] * self.conv_weights[j, k, c, f]
                conv_out[i, :, :, f] += self.conv_bias[f]
        # Apply activation (e.g., ReLU)
        conv_out = self.sigmoid(conv_out)

        # Flatten output for fully-connected layer
        flattened = conv_out.reshape(X.shape[0], -1)

        # Fully-connected layer
        fc_out = flattened.dot(self.fc_weights) + self.fc_bias

        # Apply softmax activation for probability distribution
        output = self.softmax(fc_out)
        return output

    def train(self, X, y, learning_rate=0.01, epochs=10):
        """
        Trains the CNN using gradient descent.

        Args:
          X: A numpy array representing the training images.
          y: A numpy array representing the training labels (one-hot encoded).
          learning_rate: The learning rate for gradient descent.
          epochs: The number of training epochs.
        """
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Backpropagation
            grad_output = output - y
            grad_fc_weights = np.dot(flattened.T, grad_output)
            grad_fc_bias = np.sum(grad_output, axis=0, keepdims=True)
            
            grad_fc_input = np.dot(grad_output, self.fc_weights.T)
            grad_conv_input = grad_fc_input.reshape(conv_out.shape) * (conv_out * (1 - conv_out))
            
            grad_conv_weights = np.zeros(self.conv_weights.shape)
            grad_conv_bias = np.zeros(self.conv_bias.shape)
            
            for i in range(X.shape[0]):
                for f in range(self.num_filters):
                    for c in range(X.shape[2]):
                        for j in range(self.kernel_size[0]):
                            for k in range(self.kernel_size[1]):
                                grad_conv_weights[j, k, c, f] += np.sum(X[i, j:j + self.kernel_size[0] - 1 - k, c] * grad_conv_input[i, :, :, f])
                    grad_conv_bias[f] += np.sum(grad_conv_input[i, :, :, f])
            
            # Update weights and biases
            self.fc_weights -= learning_rate * grad_fc_weights
            self.fc_bias -= learning_rate * grad_fc_bias.ravel()
            self.conv_weights -= learning_rate * grad_conv_weights
            self.conv_bias -= learning_rate * grad_conv_bias.ravel()

            # Print loss (for monitoring)
            loss = -np.sum(y * np.log(output + 1e-8)) / X.shape[0]
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}')

# Example usage:
if __name__ == "__main__":
    # Example of using the CNN class
    input_shape = (28, 28, 1)  # Example: MNIST images shape
    num_classes = 10  # Example: Number of classes (MNIST has 10 digits)
    cnn = CNN(input_shape=input_shape, num_classes=num_classes)
    
    # Example of training the CNN
    X = np.random.randn(100, *input_shape)
    y = np.random.randn(100, num_classes)
    cnn.train(X, y)


 
