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

def decrypt_weights(weights):
    return np.array([[value.decrypt()[0] for value in weight] for weight in weights])

def decrypt_bias(bias):
    return np.array([[value.decrypt()[0] for value in weight] for weight in bias])

class BaseLayer():

    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        pass

    def backward(self, output_error, learning_rate, decrypt_model=False):
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
        self.bias = self.bias - (learning_rate * output_error)
        return input_error

class ActivationLayer(BaseLayer):

    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_error, learning_rate, decrypt_model=False):
        return self.activation_prime(self.input) * output_error

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

    def fit(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)

        for i in range(epochs):
            err = 0
            for j in range(samples):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward(output)

                err += self.loss(y_train[j], output)

                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)

            err /= samples
            if self.debug != None and i % self.debug == 0:
                print(f'epoch {i+1}/{epochs}   error={err}')
    
    def send_receive_weights(self, context):
        if self.interactive and self.encrypted_training:
            for layer in self.layers:
                if type(layer) == FullyConnectedLayer:
                    weights = [[value.decrypt()[0] for value in e_weight] for e_weight in layer.weights]
                    layer.weights = np.array([[ts.ckks_vector(context, [value]) for value in weight] for weight in weights])

                    bias = [[weight.decrypt()[0] for weight in value] for value in layer.bias]
                    layer.bias = np.array([[ts.ckks_vector(context, [weight]) for weight in value] for value in bias])

    def encrypt(self, context):
        if self.interactive and self.encrypted_training:
            for layer in self.layers:
                if type(layer) == FullyConnectedLayer:
                    layer.weights = np.array([[ts.ckks_vector(context, [value]) for value in weight] for weight in layer.weights])
                    layer.bias = np.array([[ts.ckks_vector(context, [weight]) for weight in value] for value in layer.bias])

    def decrypt(self):
       if self.interactive and self.encrypted_training:
            for layer in self.layers:
                if type(layer) == FullyConnectedLayer:
                    layer.weights = decrypt_weights(layer.weights)
                    layer.bias = decrypt_bias(layer.bias)

    def crypt_data_fit(self, x_train, y_train, epochs, learning_rate, crypt_context=None):
        samples = len(x_train)

        for i in range(epochs):
            for j in range(samples):
                output = np.array([[ts.ckks_vector(crypt_context, [value]) for value in np.squeeze(x_train[j])]])
                label = np.array([[ts.ckks_vector(crypt_context, [np.squeeze(y_train[j])])]])

                for layer in self.layers:
                    output = layer.forward(output)
                print("Forward finished")

                error = self.loss_prime(label, output)

                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate, decrypt_model=True)
                print("Backward finished")
                
            if self.debug != None and i % self.debug == 0:
                print(f"Finished epoch {i}")

        print("Decrypted weights")

    def crypt_model_fit(self, x_train, y_train, epochs, learning_rate, crypt_context=None):
        samples = len(x_train)

        self.encrypt(crypt_context)
        print("Encrypted weights")

        for i in range(epochs):
            for j in range(samples):
                output = np.array([[ts.ckks_vector(crypt_context, [value]) for value in np.squeeze(x_train[j])]])
                label = np.array([[ts.ckks_vector(crypt_context, [np.squeeze(y_train[j])])]])

                for layer in self.layers:
                    output = layer.forward(output)
                print("Forward finished")

                error = self.loss_prime(label, output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)
                print("Backward finished")
                
                self.send_receive_weights(crypt_context)
            if self.debug != None and i % self.debug == 0:
                print(f"Finished epoch {i}")

        self.decrypt()
        print("Decrypted weights")