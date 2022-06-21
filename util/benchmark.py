from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import numpy as np
import time
from heNet import recrypt, decrypt, encrypt, square, square_prime, binary_cross_entropy, binary_cross_entropy_prime
from heNet import BaseLayer, EncryptedActivationLayer, EncryptedFullyConnectedLayer, Network, FullyConnectedLayer, ActivationLayer
import tenseal as ts

class Benchmark:

    def __init__(self):
        iris = load_iris()
        X = iris['data']
        y = iris['target']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=2)

        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)

        def transform(data):
            return np.array([[row] for row in data])

        self.X_train = transform(X_train)
        self.X_test = transform(X_test)
        self.y_train = transform(y_train)
        self.y_test = transform(y_test)

    def get_net(self):
        net = Network(debug=None)
        net.add(FullyConnectedLayer(4, 4))
        net.add(ActivationLayer(square, square_prime))
        net.add(FullyConnectedLayer(4, 3))
        net.add(ActivationLayer(square, square_prime))
        net.use(binary_cross_entropy, binary_cross_entropy_prime)
        return net

    def create_normale_net(self):
        net = self.get_net()
        net.fit(self.X_train, self.y_train, epochs=20, learning_rate=0.002, minibatch=8)
        return net

    def normal_inference_test(self, net, data):
        start = time.time_ns()
        net.predict(data)
        end = time.time_ns()
        return (len(data), end-start)

    def he_inference_test(self, net, data):
        poly_mod_degree = 2 ** 13
        bits_scale = 26
        coeff_mod_bit_sizes=[40, bits_scale, bits_scale, bits_scale, bits_scale, 40]
        context = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
        context.global_scale = pow(2, bits_scale)
        context.generate_galois_keys()

        e_test = np.array([[[ts.ckks_vector(context, [value]) for value in row] for row in matrix] for matrix in data])
        start = time.time_ns()
        net.predict(e_test)
        end = time.time_ns()
        return (len(data), end-start)

    def benchmark_inference(self):
        testNet = self.create_normale_net()
        test_length_list = [100,250,500]
        for size in test_length_list:
            data = [self.X_train[0]] * size
            normal_inference_result = self.normal_inference_test(testNet,data)
            print("Normal Inference: ",normal_inference_result)
            he_inference_result = self.he_inference_test(testNet,data)
            print("HE Inference: ",he_inference_result)
    
    def he_training_test(self, epochs, learning_rate, minibatch):
        poly_mod_degree = 2 ** 13
        bits_scale = 40

        coeff_mod_bit_sizes=[60, bits_scale, 60]
        context = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)

        context.global_scale = pow(2, bits_scale)
        context.generate_galois_keys()
        net = Network()
        net.add(EncryptedFullyConnectedLayer(4, 4, context))
        net.add(EncryptedActivationLayer(square, square_prime, context))
        net.add(EncryptedFullyConnectedLayer(4, 3, context))
        net.add(EncryptedActivationLayer(square, square_prime, context))

        net.use(binary_cross_entropy, binary_cross_entropy_prime)

        X_train = np.array([[[ts.ckks_vector(context, [value]) for value in row] for row in matrix] for matrix in self.X_train])
        y_train = np.array([[[ts.ckks_vector(context, [value]) for value in row] for row in matrix] for matrix in self.y_train])

        start = time.time_ns()
        net.benchmark_crypt_fit(X_train, y_train, epochs=epochs, learning_rate=learning_rate, minibatch=minibatch, context=context)
        end = time.time_ns()
        
        return end-start

    def normal_training_test(self, epochs, learning_rate, minibatch):
        net = self.get_net()

        start = time.time_ns()
        net.fit(self.X_train, self.y_train, epochs=epochs, learning_rate=learning_rate, minibatch=minibatch,)
        end = time.time_ns()

        return end-start

    def benchmark_training(self, epochs, learning_rate, minibatch):
        print(f"Epochs: {epochs} - Learning rate: {learning_rate} - Minibatch: {minibatch}")
        normal_result = self.normal_training_test(epochs,learning_rate,minibatch)
        print("Normal Training: ", normal_result)
        he_result = self.he_training_test(epochs,learning_rate,minibatch)
        print("HE Training: ", he_result)