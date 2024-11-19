from parse_args import parse_args_train
from load import load_csv
from weights_initializer import weights_biases_initialize
import numpy as np
# from pandas as pd

epsilon = 1e-15 # ε（イプシロン）はゼロ割を防ぐために加える小さな値

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # オーバーフロー防止
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def compute_output_gradients(output_node, batch_label):
    return output_node - batch_label

def compute_hidden_gradients(hidden_layers, weights, output_gradients):
    gradients = []
    # grad = np.dot(output_gradients, weights[-1]) * sigmoid_derivative(hidden_layers[-1])
    # gradients.append(grad)
    # # print(grad.shape)
    # for i in reversed(range(len(hidden_layers))):
    #     grad = np.dot(grad, weights[i]) * sigmoid_derivative(hidden_layers[i])
    #     gradients.append(grad)
    #     print(grad.shape)
    # gradients.reverse()
    # print(len(gradients))
    grad = 
    return gradients

def load_data(args):
    df_train = load_csv(args.file_train)
    df_test = load_csv(args.file_test)
    return df_train, df_test

def split_data(data):
    df_shuffled = data.sample(frac=1).reset_index(drop=True)
    data_array = df_shuffled.iloc[:, 1:].to_numpy()
    labels = df_shuffled.iloc[:,0].to_numpy()
    data_label = np.array([[1, 0] if label == 'M' else [0, 1] for label in labels])
    return data_array, data_label

# def estimate(data, weights, biases):
#     """もうこの関数使わないでいいかも 下の関数の２つ目がこの関数のreturn になる"""
#     results = sigmoid(np.dot(data, weights[0].T) + biases[0])
#     for layer_index in range(1, len(weights)):
#         results = sigmoid(np.dot(results, weights[layer_index].T) + biases[layer_index])
#     return results

def forward_propagation(batch_array, weights, biases):
    hidden_layers = []
    results = sigmoid(np.dot(batch_array, weights[0].T) + biases[0])
    hidden_layers.append(results) 
    for layer_index in range(1, len(weights) -1):
        results = sigmoid(np.dot(results, weights[layer_index].T) + biases[layer_index])
        hidden_layers.append(results)
    output_node = softmax(np.dot(results, weights[-1].T) + biases[-1])

    # results = sigmoid(np.dot(results, weights[-1].T) + biases[-1])
    # hidden_layers.append(results)

    return hidden_layers, output_node

def update_weights_and_biases(weights, biases, hidden_layers, gradients, output_gradients, learning_rate):
    weights[-1] -= learning_rate * np.dot(output_gradients.T, hidden_layers[-1])
    biases[-1] -= learning_rate * np.sum(output_gradients, axis=0, keepdims=True)
    for i in range(len(weights) - 1, -1, -1):
        # print(hidden_layers[i].shape)
        # print(gradients[i].shape)
        # print(weights[i].shape)
        weights[i] -= learning_rate * np.dot(hidden_layers[i].T, gradients[i])
        biases[i] -= learning_rate * np.sum(gradients[i], axis=0, keepdims=True)
    # return weights, biases

def main():
    args = parse_args_train()
    # print(args)
    df_train , df_test = load_data(args)
    print(f"x_train shape : {df_train.shape}")
    print(f"x_valid shape : {df_test.shape}")
    test_array, test_label = split_data(df_test)
    weights, biases =  weights_biases_initialize(df_test.shape[1] - 1, args.layer, args.weights_initializer)
    # print(test_array)
    # print(test_label)
    # print(train_array)
    # print(train_label)
    # print(train_array[0])
    # print(weights[0][0])
    # results = np.zeros(2)
    # print(train_array.shape[0])

    for epochs in range(args.epochs):
        train_array , train_label = split_data(df_train)
        for batch_start in range(0, len(train_array), args.batch_size):
            batch_array = train_array[batch_start: batch_start + args.batch_size]
            batch_label = train_label[batch_start: batch_start + args.batch_size]
            # batch_estimate = estimate(batch_array, weights, biases)
            hidden_node, output_node = forward_propagation(batch_array, weights, biases)#hidden_node -> list output_node->numpy
            # output_gradients = output_node - batch_label
            gradients = compute_hidden_gradients(hidden_node, weights, output_node)
            # update_weights_and_biases(weights, biases, hidden_node, gradients,
            #                           output_gradients, args.learning_rate)
            # loss = -np.mean(np.sum(batch_label * np.log(batch_estimate + 1e-7), axis=1))#0値処理のために1e-7を足してる
            # print(loss)
        # results = estimate(train_array, weights)
        # print(weights)
    # print(results - train_lavel)


if __name__ == "__main__":
    main()