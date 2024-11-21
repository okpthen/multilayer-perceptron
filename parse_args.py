import argparse

layer =[24,24,24]
epochs = 84
batch_size = 8
learning_rate = 0.01
split_rate = [8, 2] # train vs test
file_path = "dataset/data.csv"
loss = "binaryCrossentropy"
loss_choices=["binaryCrossentropy"]
weights_initializer = 'heUniform'
weights_initializer_choices = ['heUniform']
file_train = "dataset/train.csv"
file_test = "dataset/test.csv"
file_weights = "dataset/weights.pkl"
file_biases = "dataset/baiases.pkl"

def parse_args_train():
    parser = argparse.ArgumentParser(description='Machine Learning project\tMultilayer Perceptron')
    parser.add_argument('--layer', type=int,  nargs='+', default=layer, help="Sizes of the hidden layers")
    parser.add_argument('--epochs', type=int, default=epochs, help="Number of epochs")
    parser.add_argument('--loss', type=str, default=loss, help='Loss Function', choices=loss_choices) #ここらへんは一応書いておいただけ
    parser.add_argument('--weights_initializer', type=str, default=weights_initializer, 
                        choices=weights_initializer_choices, help='weights_initializer') #ここらへんは一応書いておいただけ
    parser.add_argument('--batch_size', type=int, default=batch_size, help="Batch size")
    parser.add_argument('--learning_rate', type=float, default=learning_rate, help="Learning rate")
    parser.add_argument('--file_train', type=str, default=file_train, help='train file path')
    parser.add_argument('--file_test', type=str, default=file_test, help='test file path')
    return parser.parse_args()


def parse_args_prepare():
    parser = argparse.ArgumentParser(description='Machine Learning project\tMultilayer Perceptron')
    parser.add_argument('--file', type=str, default=file_path, help='file path')
    parser.add_argument('--rate', type=int, nargs=2, default=split_rate, help='train vc test data rate')
    return parser.parse_args()