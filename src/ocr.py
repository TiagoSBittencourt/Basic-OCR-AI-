import numpy as np
import os
import json

class OCRNeuralNetwork:
    NN_FILE_PATH = os.path.join(os.path.dirname(__file__), "neural_network.json")
    LEARNING_RATE = 0.1  # Exemplo, ajuste conforme necessário

    def __init__(self, num_hidden_nodes, data_matrix, data_labels, train_indices, use_file=True):
        self._use_file = use_file
        if self._use_file and os.path.exists(OCRNeuralNetwork.NN_FILE_PATH):
            self._load()
        else: 
            self.theta1 = self._rand_initialize_weights(400, num_hidden_nodes)
            self.theta2 = self._rand_initialize_weights(num_hidden_nodes, 10)
            self.input_layer_bias = self._rand_initialize_weights(1, num_hidden_nodes)
            self.hidden_layer_bias = self._rand_initialize_weights(1, 10)
        

    def _rand_initialize_weights(self, size_in, size_out):
        return (np.random.rand(size_out, size_in) * 0.12) - 0.06

    def _sigmoid_scalar(self, z):
        return 1 / (1 + np.exp(-z))
    
    def relu(self, x):
        return np.maximum(0, x)

    def relu_prime(self, x):
        return (x > 0).astype(float)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z):
        sig = self.sigmoid(z)
        return sig * (1 - sig)
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z))  
        return exp_z / np.sum(exp_z)

    def train_on_instance(self, data):
        self.input_layer_bias = self.input_layer_bias.flatten()
        self.hidden_layer_bias = self.hidden_layer_bias.flatten()

        y1 = np.dot(self.theta1, np.array(data['y0']).T)
        sum1 = y1 + self.input_layer_bias  # shape: (16,)
        y1 = self.relu(sum1)

        y2 = np.dot(self.theta2, y1)  # shape: (10,)
        y2 = y2 + self.hidden_layer_bias
        y2 = self.softmax(y2)

        actual_vals = [0] * 10
        actual_vals[data['label']] = 1

        output_errors = np.array(actual_vals).reshape(-1, 1) - y2.reshape(-1, 1)
        hidden_errors = np.multiply(np.dot(self.theta2.T, output_errors), self.relu_prime(sum1).reshape(-1, 1))

        self.theta1 += self.LEARNING_RATE * np.dot(hidden_errors, np.array(data['y0']).reshape(1, -1))
        self.theta2 += self.LEARNING_RATE * np.dot(output_errors, y1.reshape(1, -1))
        self.hidden_layer_bias += (self.LEARNING_RATE * output_errors).flatten()
        self.input_layer_bias += (self.LEARNING_RATE * hidden_errors).flatten()


    def predict(self, test):
        y1 = np.dot(self.theta1, np.array(test).T)
        y1 = y1 + np.array(self.input_layer_bias)
        y1 = self.relu(y1)
        y2 = np.dot(self.theta2, y1)
        y2 = y2 + np.array(self.hidden_layer_bias)
        y2 = self.softmax(y2)
        return int(np.argmax(y2))

    def save(self):
        if not self._use_file:
            return

        json_neural_network = {
            "theta1": self.theta1.tolist(),
            "theta2": self.theta2.tolist(),
            "b1": self.input_layer_bias.tolist(),
            "b2": self.hidden_layer_bias.tolist()
        }
        with open(OCRNeuralNetwork.NN_FILE_PATH, 'w') as nnFile:
            json.dump(json_neural_network, nnFile)

    def _load(self):
        if not self._use_file:
            return
        with open(OCRNeuralNetwork.NN_FILE_PATH) as nnFile:
            nn = json.load(nnFile)
        self.theta1 = np.array(nn['theta1'])
        self.theta2 = np.array(nn['theta2'])
        self.input_layer_bias = np.array(nn['b1'])
        self.hidden_layer_bias = np.array(nn['b2'])