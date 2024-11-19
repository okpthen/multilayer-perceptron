import pandas as pd
import sys

def load_csv(path: str):
    try:
        df = pd.read_csv(path, header=None)
    except FileNotFoundError:
        print(f"Error: The file at {path} was not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        sys.exit(1)
    except pd.errors.ParserError:
        print(f"Error: The file at {path} could not be parsed. It may not be a valid CSV file.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
    # print("Loading dataset of dimensions", df.shape)
    print(f"Loading dataset {path} srcsess")
    return df
