import glob
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def extract_features(df, sample_rate=512):
    """
    Chunks the continuous CSV data into 1-second windows, filters out bad signals,
    and calculates the Engagement Index (EI) and band averages.
    """
    features = []

    # Calculate how many full 1-second chunks we have
    num_chunks = len(df) // sample_rate

    for i in range(num_chunks):
        # Extract the 1-second window (512 rows)
        start_idx = i * sample_rate
        end_idx = start_idx + sample_rate
        window = df.iloc[start_idx:end_idx]

        # --- NOISE FILTER ---
        # If the headset lost connection at any point in this second, throw the window away.
        if window["poorSignal"].max() > 100:
            continue

        # Calculate the average power for each band over this 1-second window
        theta = window["theta"].mean()
        alpha = window["alphaLow"].mean() + window["alphaHigh"].mean()
        beta = window["betaLow"].mean() + window["betaHigh"].mean()
        gamma = window["gammaLow"].mean() + window["gammaMid"].mean()

        # Calculate the Engagement Index (EI)
        # Add a tiny number (1e-6) to the denominator to prevent division by zero
        ei = beta / (alpha + theta + 1e-6)

        # Store all features for this window
        # We use multiple features because EI combined with raw bands is much more accurate
        window_features = [theta, alpha, beta, gamma, ei]
        features.append(window_features)

    return features


def build_and_train_pipeline(input_dir, model_save_path="anxiety_rf_model.pkl"):
    """
    Reads all data, extracts features, trains the Random Forest, and evaluates it.
    """
    X_list = []
    y_list = []

    # Map folders to integer labels
    label_map = {
        "level_1": 0,  # Normal
        "level_2": 1,  # Light
        "level_3": 2,  # Moderate
        "level_4": 3,  # Severe
    }

    print("--- Phase 1: Data Processing & Feature Extraction ---")
    for folder_name, numeric_label in label_map.items():
        folder_path = os.path.join(input_dir, folder_name)

        if not os.path.exists(folder_path):
            print(f"Warning: {folder_name} not found. Skipping.")
            continue

        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        folder_features = []

        for file in csv_files:
            try:
                df = pd.read_csv(file)
                # Extract the 1-second features
                extracted = extract_features(df)
                folder_features.extend(extracted)
            except Exception as e:
                print(f"Failed to process {file}: {e}")

        if folder_features:
            X_list.extend(folder_features)
            # Create matching labels
            y_list.extend([numeric_label] * len(folder_features))
            print(
                f"Processed {folder_name}: Extracted {len(folder_features)} clean 1-second windows."
            )

    # Convert to Numpy Arrays for Scikit-Learn
    X = np.array(X_list)
    y = np.array(y_list)

    if len(X) == 0:
        print("Error: No valid data found. Check folder names and poorSignal values.")
        return

    print(f"\nTotal Dataset: {X.shape[0]} samples across {X.shape[1]} features.")

    print("\n--- Phase 2: Machine Learning ---")
    # Split the data: 80% for training the model, 20% to test its accuracy
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples...")

    # Initialize the Random Forest Classifier
    # n_estimators=100 means it uses 100 decision trees to vote on the outcome
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

    # Train the model
    clf.fit(X_train, y_train)

    # Test the model on the 20% of data it hasn't seen yet
    y_pred = clf.predict(X_test)

    # Evaluate the results
    acc = accuracy_score(y_test, y_pred)
    print(f"\nOverall Model Accuracy: {acc * 100:.2f}%")
    print("\nDetailed Classification Report:")

    # Define target names based on the classes that actually exist in BOTH test and pred
    present_classes = np.unique(np.concatenate((y_test, y_pred)))
    target_names = [list(label_map.keys())[i] for i in present_classes]

    # Pass the explicit labels array to the report generator to prevent mismatches
    print(
        classification_report(
            y_test, y_pred, labels=present_classes, target_names=target_names
        )
    )
    # Save the trained model to a file
    joblib.dump(clf, model_save_path)
    print(f"\nModel saved successfully as: {model_save_path}")
    print("You can load this file later to make live predictions!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Anxiety ML Model")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="./raw_data",
        help="Path to parent directory containing level folders",
    )
    args = parser.parse_args()

    build_and_train_pipeline(args.input)
