import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from heNet import square, square_prime, sigmoid, sigmoid_prime, binary_cross_entropy, binary_cross_entropy_prime
from heNet import Network, FullyConnectedLayer, ActivationLayer

def toNp(value_list):
    return np.array([[data] for data in value_list])

def torch_train(train_index, test_index, cross_data, Net, label, minibatch):

    pytorch_y = [[label] for label in label[train_index].to_numpy()]

    train_x = torch.Tensor(cross_data[train_index]).float()
    test_x = torch.Tensor(cross_data[test_index]).float()
    train_y = torch.Tensor(pytorch_y).float()

    pytorch_net = Net()

    no_epochs = 400
    learning_rate = 0.001
    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(pytorch_net.parameters(), lr=learning_rate)

    train_losses = []
    for _ in range(0, no_epochs):
        for j in range(0, train_x.shape[0], minibatch):
            predictions = pytorch_net.forward(train_x[j:j+minibatch])

            loss = loss_func(predictions, train_y[j:j+minibatch])
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            optimizer.zero_grad()

    return pytorch_net, test_x

def he_train(train_index, cross_data, label, minibatch):
    cross_X_train = toNp(cross_data[train_index])
    cross_y_train = toNp(toNp(label[train_index].to_numpy()))
    net = Network(debug=None)
    net.add(FullyConnectedLayer(20, 10))
    net.add(ActivationLayer(square, square_prime))
    net.add(FullyConnectedLayer(10, 1))
    net.add(ActivationLayer(sigmoid, sigmoid_prime))

    net.use(binary_cross_entropy, binary_cross_entropy_prime)
    net.fit(cross_X_train, cross_y_train, epochs=400, learning_rate=0.001, minibatch=minibatch)
    
    return net

def getAcc(out, cross_y_test):
    correct = 0
    for idx,_ in enumerate(out):
        result = 1 if np.squeeze(out[idx]) > 0.5 else 0
        if result == np.squeeze(cross_y_test[idx]):
            correct = correct + 1
    return correct/len(out)

def cross_validate_he_torch(he_net, torch, heart_dataframe, label):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    scaler = MinMaxScaler()
    cross_data = scaler.fit_transform(heart_dataframe)
    he_acc_sum = []
    torch_acc_sum = []
    for train_index, test_index in kf.split(cross_data):
        cross_X_test = toNp(cross_data[test_index])
        cross_y_test =  toNp(label[test_index].to_numpy())

        cross_he_net = he_train(train_index, cross_data, label, 8)
        he_out = cross_he_net.predict(cross_X_test)
        he_acc_sum.append(getAcc(he_out, cross_y_test))

        torch_net, torch_X_test = torch_train(train_index, test_index, cross_data, torch, label, 8)
        torch_out = torch_net(torch_X_test).detach().numpy()
        torch_acc_sum.append(getAcc(torch_out, cross_y_test))
    return he_acc_sum, torch_acc_sum