import math
import numpy as np

def heUniform(input :int, layer:list):
    weights = []
    fan_in = input
    layer.append(2)
    for neurons in layer:
        limits = math.sqrt(6 / fan_in)
        layer_weights = np.random.uniform(low=-limits, high=limits, 
                                          size=(neurons, fan_in))
        weights.append(layer_weights)
        fan_in = neurons
        # print(layer_weights.shape[0], layer_weights.shape[1])
    return weights

def heBiases(layer:list):
    biases = []
    for neurons in layer:
        biases.append(np.zeros((1, neurons)))
    biases.append(np.zeros((1, 2)))
    return biases


def weights_biases_initialize(input :int, layer:list, inirializer:str):
    biases = heBiases(layer)
    if inirializer == "heUniform":
        return heUniform(input, layer), biases
    else:
        print("def weights_initialize error")