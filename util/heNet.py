import numpy as np
import tenseal as ts

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

def square(x):
    return np.multiply(x,x)

def square_prime(x):
    return np.multiply(2, x)

def approx_sigmoid(x):
    return 0.5 + 0.197 * x - 0.0004 * (x**3)

def approx_sigmoid_prime(x):
    return approx_sigmoid(x)*(1 - approx_sigmoid(x))

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

def binary_cross_entropy(y_true, y_pred):
    return y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)

def binary_cross_entropy_prime(y_true, y_pred):
    return y_pred - y_true

def cross_entropy(y_true, y_pred):
    return -sum([y_true[i]*np.log(y_pred[i]) for i in range(len(y_true))])

def cross_entropy_prime(y_true, y_pred):
    return y_pred - y_true

def encrypt(matrix, context):
    return np.array([[ts.ckks_vector(context, [value]) for value in column] for column in matrix])

def encrypt_Label(vector, context):
    return np.array([ts.ckks_vector(context, [value]) for value in vector])

def decrypt(matrix):
    return np.array([[column.decrypt()[0] for column in row] for row in matrix])

def recrypt(matrix, context):
    e_value = decrypt(matrix)
    return encrypt(e_value, context)

class BaseLayer():

    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        pass

    def backward(self, output_error, learning_rate):
        pass

class FullyConnectedLayer(BaseLayer):

    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        self.weights = self.weights - (learning_rate * weights_error)
        self.bias = self.bias - (learning_rate * np.sum(output_error, axis=0, keepdims=True))
        return input_error

class EncryptedFullyConnectedLayer(BaseLayer):

    def __init__(self, input_size, output_size, context):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
        self.context = context

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def encrypted_forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return recrypt(self.output,self.context)

    def backward(self, output_error, learning_rate):
        weights_error = np.dot(self.input.T, output_error)
        weights_error = recrypt(weights_error,self.context)

        self.weights = self.weights - (learning_rate * weights_error)
        self.weights = recrypt(self.weights,self.context)
   
        self.bias = self.bias - (learning_rate * np.sum(output_error, axis=0, keepdims=True))
        self.bias = recrypt(self.bias,self.context)

        input_error = np.dot(output_error, self.weights.T)
        return recrypt(input_error,self.context)

class ActivationLayer(BaseLayer):

    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error

class EncryptedActivationLayer(BaseLayer):

    def __init__(self, activation, activation_prime, context):
        self.activation = activation
        self.activation_prime = activation_prime
        self.context = context

    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def encrypted_forward(self, input_data):
        self.input = input_data
        output = self.activation(self.input)
        return recrypt(output,self.context)

    def backward(self, output_error, learning_rate):
        activation_error = self.activation_prime(self.input)
        output = recrypt(activation_error,self.context) * output_error
        return recrypt(output,self.context)

class Network:
    def __init__(self, debug=None, encrypted_training=False, interactive=False):
        self.layers = []
        self.loss = None
        self.loss_prime = None
        self.debug = debug
        self.encrypted_training = encrypted_training
        self.interactive = interactive

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):
        samples = len(input_data)
        result = []

        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)

        return result

    def fit(self, x_train, y_train, epochs, learning_rate, minibatch):
        samples = len(x_train)

        for i in range(epochs):
            err = 0
            for j in range(0, x_train.shape[0], minibatch):
                output = np.array([[values for values in np.squeeze(row)] for row in x_train[j:j+minibatch]])
                for layer in self.layers:
                    output = layer.forward(output)

                label = np.array([row[0] for row in y_train[j:j+minibatch]])

                error = self.loss_prime(label, output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)

            err /= samples
            if self.debug != None and i % self.debug == 0:
                print(f'epoch {i+1}/{epochs}')
    
    def crypt_fit(self, x_train, y_train, epochs, learning_rate, minibatch, context):

        for layer in self.layers:
            if type(layer) == EncryptedFullyConnectedLayer:
                layer.weights = encrypt(matrix=layer.weights, context=context)
                layer.bias = encrypt(matrix=layer.bias, context=context)

        for i in range(epochs):
            for j in range(0, x_train.shape[0], minibatch):
                output = np.array([[values for values in np.squeeze(row)] for row in x_train[j:j+minibatch]])
                output = encrypt(output,context)

                for layer in self.layers:
                    output = layer.encrypted_forward(output)

                label = np.array([row[0] for row in y_train[j:j+minibatch]])
                label = encrypt(label,context)

                error = self.loss_prime(label, output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)

            if self.debug != None and i % self.debug == 0:
                print(f'epoch {i+1}/{epochs}')

        for layer in self.layers:
            if type(layer) == EncryptedFullyConnectedLayer:
                layer.weights = decrypt(matrix=layer.weights)
                layer.bias = decrypt(matrix=layer.bias)
            
    def benchmark_crypt_fit(self, x_train, y_train, epochs, learning_rate, minibatch, context):

        for layer in self.layers:
            if type(layer) == EncryptedFullyConnectedLayer:
                layer.weights = encrypt(matrix=layer.weights, context=context)
                layer.bias = encrypt(matrix=layer.bias, context=context)

        for i in range(epochs):
            for j in range(0, x_train.shape[0], minibatch):
                output = np.array([[values for values in np.squeeze(row)] for row in x_train[j:j+minibatch]])

                for layer in self.layers:
                    output = layer.encrypted_forward(output)

                label = np.array([row[0] for row in y_train[j:j+minibatch]])

                error = self.loss_prime(label, output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)

            if self.debug != None and i % self.debug == 0:
                print(f'epoch {i+1}/{epochs}')

        for layer in self.layers:
            if type(layer) == EncryptedFullyConnectedLayer:
                layer.weights = decrypt(matrix=layer.weights)
                layer.bias = decrypt(matrix=layer.bias)