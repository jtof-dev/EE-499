import pandas as pd

# Load the dataset
dataset = pd.read_csv("sample_eeg_data.csv")

# Now you can easily isolate your variables for machine learning:
X = dataset[
    [
        "Delta",
        "Theta",
        "Alpha1",
        "Alpha2",
        "Beta1",
        "Beta2",
        "Gamma1",
        "Gamma2",
        "Attention",
        "Meditation",
    ]
]
y = dataset["label"]

print(dataset["Alpha2"].head(10))
