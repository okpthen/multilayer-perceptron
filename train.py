from parse_args import parse_args_train, file_weights, file_biases
from load import load_csv
from weights_initializer import weights_biases_initialize
import numpy as np
import pickle
# from pandas as pd

epsilon = 1e-15 # ε（イプシロン）はゼロ割を防ぐために加える小さな値
over_learn_epochs = 5
args = parse_args_train()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))  # オーバーフロー防止
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def softmax_derivative(x):
    return x

def compute_output_gradients(output_node, batch_label):
    ave = np.mean(output_node - batch_label, axis=1)
    return ave[:, np.newaxis]

def load_data():
    df_train = load_csv(args.file_train)
    df_test = load_csv(args.file_test)
    return df_train, df_test

def split_data(data):
    df_shuffled = data.sample(frac=1).reset_index(drop=True)
    data_array = df_shuffled.iloc[:, 1:].to_numpy()
    labels = df_shuffled.iloc[:,0].to_numpy()
    data_label = np.array([[1, 0] if label == 'M' else [0, 1] for label in labels])
    return data_array, data_label

def forward_propagation(batch_array, weights, biases):
    """hidden_node -> list output_node-> numpy"""
    hidden_layers = []
    results = sigmoid(np.dot(weights[0] , batch_array) + biases[0])
    hidden_layers.append(results)
    for layer_index in range(1, len(weights) -1):
        results = sigmoid(np.dot(weights[layer_index], results) + biases[layer_index])
        hidden_layers.append(results)
    output_node = softmax(np.dot(weights[-1], results) + biases[-1])
    return hidden_layers, output_node

def compute_hidden_gradients(hidden_layers, weights, output_gradients):
    gradients = []
    gradients.append(output_gradients)
    grad = output_gradients
    for i in reversed(range(len(hidden_layers))):
        grad = np.dot(weights[i + 1].T, grad)
        sigmoid_d = sigmoid_derivative(hidden_layers[i])
        sigmoid_d = np.mean(sigmoid_d, axis=1)[:, np.newaxis]
        result =  grad * sigmoid_d
        gradients.append(result)
    gradients.reverse()
    return gradients

def update_weights_and_biases(weights, biases, hidden_layers, gradients, batch_array, learning_rate):
    for i in range(len(gradients)):
        if i == 0:
            input = batch_array
        else:
            input = hidden_layers[i - 1]
        ave = np.mean(input, axis=1)[:, np.newaxis]
        weights_update =  np.dot(gradients[i] ,ave.T)
        weights[i] -= learning_rate * weights_update
        biases[i] -= learning_rate * gradients[i]
    # return weights, biases

def compute_loss_acc(array, label, weighes, biases):
    _, output_node = forward_propagation(array, weighes , biases)
    loss = -np.mean(np.sum(label * np.log(output_node + epsilon), axis=0))
    # loss = -np.mean(np.sum(label * np.log(output_node + epsilon) + (1 - label) * np.log(1 - output_node + epsilon), axis=0))
    output_node_binary = (output_node == output_node.max(axis=0))
    output_node_binary = output_node_binary.astype(int)
    accuracy = np.mean(output_node_binary == label)
    return loss ,accuracy


def main():
    # args = parse_args_train()
    df_train , df_test = load_data()
    print(f"x_train shape : {df_train.shape}")
    print(f"x_valid shape : {df_test.shape}")
    test_array, test_label = split_data(df_test)
    weights, biases =  weights_biases_initialize(df_test.shape[1] - 1, args.layer, args.weights_initializer)

    test_array = test_array.T
    test_label = test_label.T
    val_loss = 1
    over_learning_count = 0

    for epochs in range(args.epochs):
        train_array , train_label = split_data(df_train)
        for batch_start in range(0, len(train_array), args.batch_size):
            batch_array = train_array[batch_start: batch_start + args.batch_size].T
            batch_label = train_label[batch_start: batch_start + args.batch_size].T
            hidden_node, output_node = forward_propagation(batch_array, weights, biases)
            output_gradient = compute_output_gradients(output_node, batch_label) #損失関数とシグモイドを足したのの微分
            gradients = compute_hidden_gradients(hidden_node, weights, output_gradient)
            update_weights_and_biases(weights, biases, hidden_node, gradients,
                                      batch_array, args.learning_rate)
        loss, train_accuracy = compute_loss_acc(train_array.T, train_label.T, weights, biases)
        pre_loss = val_loss
        val_loss , test_accuracy = compute_loss_acc(test_array, test_label, weights, biases)

        print(f"epoch {epochs + 1}/{args.epochs} - loss: {loss:.4f} - val_loss: {val_loss:.4f} - accuracy:{train_accuracy:.4f} - val_accuracy: {test_accuracy:.4f}")
        # print(f"epoch {epochs + 1}/{args.epochs} - loss: {loss:.4f} - val_loss: {val_loss:.4f}")
        if (pre_loss < val_loss):
            over_learning_count += 1
            if (over_learning_count == over_learn_epochs):
                print("over leaning")
                break
        else:
            tmp_weight = weights
            tmp_biase = biases
            over_learning_count = 0

    with open(file_weights, 'wb') as f:
        pickle.dump(tmp_weight, f)
    with open(file_biases, 'wb') as f:
        pickle.dump(tmp_biase, f)
    


if __name__ == "__main__":
    main()