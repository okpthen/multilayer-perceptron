from parse_args import parse_args_prepare, file_test, file_train
from load import load_csv
import pandas as pd

def main():
    args = parse_args_prepare()
    # print(args.rate)
    df = load_csv(args.file)
    # print(df)
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
    # print(std_data)
    df_shuffled = std_data.sample(frac=1).reset_index(drop=True)
    # print(df_shuffled)
    train_size = int((args.rate[0] / (args.rate[0] + args.rate[1])) * len(df_shuffled))
    train_data = df_shuffled[:train_size]
    test_data = df_shuffled[train_size:]
    train_data.to_csv(file_train, index=False, header=False)
    test_data.to_csv(file_test, index=False, header=False)


if __name__ == "__main__":
    main()