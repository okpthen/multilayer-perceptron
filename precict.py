import pickle
from parse_args import  file_weights, file_biases, file_path
from load import load_csv
from train import split_data, compute_loss_acc
import pandas as pd

def prepare_data(df):
    df.dropna()
    std_data = pd.DataFrame()
    std_data["diagnosis"] = df.iloc[:,1]
    index = 1
    for col_index in range(2, df.shape[1]):
        data = df.iloc[:, col_index]
        std = data.std()
        mean = data.mean()
        standard =(data - mean)  / std
        std_data[index] = standard
        index += 1
    return std_data

def main():
    with open(file_weights, 'rb') as f:
        weights = pickle.load(f)
    with open(file_biases, 'rb') as f:
        biases = pickle.load(f)
    df = load_csv(file_path)
    df = prepare_data(df)
    array, labesl = split_data(df)
    array = array.T
    labesl = labesl.T
    loss, acc = compute_loss_acc(array, labesl, weights, biases)
    print(f"loss = {loss:.4f} accuracy = {acc:.4f}")

if __name__ == "__main__":
    main()