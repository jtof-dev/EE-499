import pandas as pd

# load the dataset
dataset = pd.read_csv("sample_eeg_data.csv")

print(dataset["Alpha2"].head(10))
