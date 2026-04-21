import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import signal

# Official SingLEM Imports
from SingLEM.model import Config, EEGEncoder
from torch.utils.data import DataLoader, Dataset

# --- CONFIGURATION ---
DATA_BASE_DIR = "data"
PRETRAINED_WEIGHTS_PATH = "weights/singlem_pretrained.pt"

# Pulling directly from your sampleData.csv
EEG_COLUMN = "eegRawValueVolts"

# Signal Processing Variables
ORIGINAL_FS = 512
TARGET_FS = 128
DOWNSAMPLE_FACTOR = ORIGINAL_FS // TARGET_FS  # 4

WINDOW_SECONDS = 5
TOKENS_PER_WINDOW = WINDOW_SECONDS  # 5 tokens
SAMPLES_PER_TOKEN = TARGET_FS  # 128 samples

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4

FILENAME_REGEX = re.compile(
    r"^\d{8}_\d{4}_[A-Za-z0-9]+_EEG_(?P<test>[A-Za-z0-9]+)_(?P<condition>[A-Za-z0-9]+)\.csv$",
    re.IGNORECASE,
)


def preprocess_eeg(raw_volts, fs=ORIGINAL_FS):
    """Applies the strict SingLEM preprocessing pipeline."""
    # 1. Bandpass filter (0.5 Hz to 50 Hz)
    # 4th order Butterworth filter
    b, a = signal.butter(4, [0.5, 50.0], btype="bandpass", fs=fs)
    filtered = signal.filtfilt(b, a, raw_volts)

    # 2. Notch filter (60 Hz for US power-line interference)
    b_notch, a_notch = signal.iirnotch(60.0, 30.0, fs=fs)
    filtered = signal.filtfilt(b_notch, a_notch, filtered)

    # 3. Resample to 128 Hz (decimate includes anti-aliasing)
    resampled = signal.decimate(filtered, q=DOWNSAMPLE_FACTOR)

    # 4. Amplitude Scaling (Volts * 10,000)
    scaled = resampled * 1e4

    return scaled


class EEGLevelDataset(Dataset):
    def __init__(self, base_dir):
        self.samples = []
        self.labels = []

        print("Crawling dataset directories...")
        for level in range(1, 5):
            folder_path = os.path.join(base_dir, f"level_{level}")
            if not os.path.exists(folder_path):
                continue

            pytorch_label = level - 1

            for filename in os.listdir(folder_path):
                match = FILENAME_REGEX.match(filename)
                if match:
                    filepath = os.path.join(folder_path, filename)
                    self._process_file(filepath, pytorch_label)

        print(
            f"Dataset compiled: {len(self.samples)} total {WINDOW_SECONDS}-second windows."
        )

    def _process_file(self, filepath, label):
        try:
            df = pd.read_csv(filepath)
            if EEG_COLUMN not in df.columns:
                return

            # Extract raw volts from the CSV
            raw_volts = df[EEG_COLUMN].values

            # Apply the official preprocessing pipeline
            processed_eeg = preprocess_eeg(raw_volts, fs=ORIGINAL_FS)

            # Slice into Windows
            window_size = TOKENS_PER_WINDOW * SAMPLES_PER_TOKEN  # 5 * 128 = 640 samples
            step_size = SAMPLES_PER_TOKEN  # Slide by 1 second (128 samples)

            for start_idx in range(0, len(processed_eeg) - window_size, step_size):
                window = processed_eeg[start_idx : start_idx + window_size]

                # Reshape to SingLEM format: (num_tokens, samples_per_token) -> (5, 128)
                reshaped_window = window.reshape(TOKENS_PER_WINDOW, SAMPLES_PER_TOKEN)

                self.samples.append(reshaped_window)
                self.labels.append(label)

        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class AnxietyClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(AnxietyClassifier, self).__init__()

        # 1. Initialize SingLEM Feature Extractor
        config = Config()
        config.mask_prob = 0.0
        self.feature_extractor = EEGEncoder(config)

        # 2. Load Pretrained Weights
        if not os.path.exists(PRETRAINED_WEIGHTS_PATH):
            raise FileNotFoundError(
                f"Missing pretrained weights at {PRETRAINED_WEIGHTS_PATH}."
            )

        encoder_state = torch.load(PRETRAINED_WEIGHTS_PATH, map_location="cpu")
        self.feature_extractor.load_state_dict(encoder_state)

        # 3. Freeze the Feature Extractor (as dictated by the paper)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # 4. Custom Classification Head
        # SingLEM outputs (batch, num_tokens, hidden_dim).
        # Hidden dim is 16. So 5 tokens * 16 = 80 flattened features.
        flattened_size = TOKENS_PER_WINDOW * 16

        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x shape: (batch_size, 5, 128)
        features, _, _ = self.feature_extractor(x)
        # features shape: (batch_size, 5, 16)

        # Flatten the features for the dense layers
        features = features.view(features.size(0), -1)
        # flattened shape: (batch_size, 80)

        logits = self.classifier(features)
        return logits


def train_model():
    dataset = EEGLevelDataset(DATA_BASE_DIR)
    if len(dataset) == 0:
        print("No valid EEG data found. Exiting.")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = AnxietyClassifier(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()

    # Only train our custom classifier head
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

    print("Beginning Training Loop...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)

            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += batch_y.size(0)
            correct_predictions += (predicted == batch_y).sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = (correct_predictions / total_predictions) * 100
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%"
        )

    print("Training complete. Saving weights...")
    torch.save(model.state_dict(), "singlem_anxiety_head.pth")


if __name__ == "__main__":
    train_model()

