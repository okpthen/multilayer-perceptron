import pickle
from parse_args import  file_weights, file_biases

def main():
    with open(file_weights, 'rb') as f:
        weights = pickle.load(f)
    with open(file_biases, 'rb') as f:
        biases = pickle.load(f)
    print(weights[0].shape)
    print(weights[1].shape)
    print(weights[2].shape)
    print(weights[3].shape)
    print(biases[0].shape)
    print(biases[1].shape)
    print(biases[2].shape)
    print(biases[3].shape)

if __name__ == "__main__":
    main()