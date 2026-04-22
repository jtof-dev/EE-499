import os
import re
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import signal

# import SingLEM from git submodule
sys.path.append(os.path.join(os.path.dirname(__file__), "SingLEM"))

from SingLEM.model import Config, EEGEncoder
from torch.utils.data import DataLoader, Dataset

# configuration
DATA_BASE_DIR = "data"
PRETRAINED_WEIGHTS_PATH = "SingLEM/weights/singlem_pretrained.pt"
EEG_COLUMN = "eegRawValueVolts"

BATCH_SIZE = 32
EPOCHS = 50
DISCARD_SECONDS = 60

# set to None to include all participants, or use a regex string like r"P0[1-5]"
TARGET_PARTICIPANT_REGEX = r"^Andy$"


def preprocess_eeg(raw_volts):
    """
    SingLEM 128Hz preprocessing pipeline.
    Applies a bandpass filter (0.5-50Hz) to isolate brainwaves, a notch filter (60Hz)
    to remove electrical line noise, downsamples to 128Hz, and normalizes the signal.
    """
    # bandpass filter (0.5Hz - 50Hz)
    b, a = signal.butter(4, [0.5, 50.0], btype="bandpass", fs=512)
    filtered = signal.filtfilt(b, a, raw_volts)

    # notch filter (60Hz)
    b_notch, a_notch = signal.iirnotch(60.0, 30.0, fs=512)
    filtered = signal.filtfilt(b_notch, a_notch, filtered)

    # downsample (512Hz -> 128Hz); q=4 downsampling factor (512 / 4 = 128)
    resampled = signal.decimate(filtered, q=4)
    scaled = resampled * 1e4  # scale to [-1, 1] typical range for neural networks

    # z-score normalization
    normalized = (scaled - np.mean(scaled)) / (np.std(scaled) + 1e-8)

    return normalized


class BinaryEEGDataset(Dataset):
    def __init__(self, mode="train", participant_regex=None):
        self.samples = []
        self.labels = []
        self.mode = mode

        # mapping folder structures to binary classes
        # 0 = low anxiety, 1 = high anxiety
        level_mapping = {1: 0, 2: 1}

        print(f"Loading Silent EEG data for {mode.upper()}...")

        # compile regex once for performance if provided
        regex_pattern = re.compile(participant_regex) if participant_regex else None

        for level, py_class in level_mapping.items():
            folder = os.path.join(DATA_BASE_DIR, f"level_{level}")
            if not os.path.exists(folder):
                continue

            for file in os.listdir(folder):
                if not file.endswith(".csv"):
                    continue

                # expected format: YYYYMMDD_HHMM_PARTICIPANT_DATATYPE_TEST_TESTCONDITION.csv
                parts = file.replace(".csv", "").split("_")

                # ensure the filename actually conforms to the convention before parsing
                if len(parts) >= 6:
                    participant_name = parts[2]
                    datatype = parts[3]
                    test_condition = parts[5]
                else:
                    continue  # skip files that don't match the expected naming structure

                # apply filters
                # condition and datatype filter
                if datatype != "EEG" or test_condition != "Silent":
                    continue

                # participant regex filter
                if regex_pattern and not regex_pattern.search(participant_name):
                    continue

                path = os.path.join(folder, file)
                df = pd.read_csv(path)

                # ensure required columns exist before processing
                if EEG_COLUMN not in df.columns or "poorSignal" not in df.columns:
                    continue

                raw_volts = df[EEG_COLUMN].values
                poor_signal = df["poorSignal"].values

                # settling logic
                samples_to_discard = DISCARD_SECONDS * 512

                if len(raw_volts) <= samples_to_discard:
                    if (
                        mode == "train"
                    ):  # prevent console spam by warning only on train pass
                        print(
                            f"warning: {file} is shorter than {DISCARD_SECONDS}s; skipping"
                        )
                    continue

                # slice off the first minute of data from both arrays
                raw_volts = raw_volts[samples_to_discard:]
                poor_signal = poor_signal[samples_to_discard:]

                # preprocess the whole continuous array at once to prevent edge-effect glitches that occur if you filter piece-by-piece
                processed_eeg = preprocess_eeg(raw_volts)

                # chronological split
                # 80% for training, 20% for validation
                # splitting sequentially instead of randomly prevents highly correlated adjacent windows from bleeding across the train/val barrier.
                split_idx_128 = int(len(processed_eeg) * 0.8)
                split_idx_512 = int(len(poor_signal) * 0.8)

                if mode == "train":
                    processed_eeg = processed_eeg[:split_idx_128]
                    poor_signal = poor_signal[:split_idx_512]
                elif mode == "val":
                    processed_eeg = processed_eeg[split_idx_128:]
                    poor_signal = poor_signal[split_idx_512:]

                # window rejection logic
                window_size_128 = 5 * 128
                step_size_128 = 128  # slide by 1s (overlapping windows)

                # slide through the 128Hz array
                for start_128 in range(
                    0, len(processed_eeg) - window_size_128, step_size_128
                ):
                    start_512 = start_128 * 4
                    window_size_512 = window_size_128 * 4  # 2560 samples

                    signal_window = poor_signal[start_512 : start_512 + window_size_512]

                    # if the headset lost connection/dropped quality at ANY ms in this window, throw it out
                    if np.any(signal_window > 45):
                        continue

                    # if the signal is perfectly clean, grab the EEG window and format it for SingLEM
                    window = processed_eeg[start_128 : start_128 + window_size_128]

                    # reshape into tokens: 5 seconds -> 5 tokens of 128 samples each
                    reshaped = window.reshape(5, 128)

                    self.samples.append(reshaped)
                    self.labels.append(py_class)

        print(f"loaded {len(self.samples)} clean {mode.upper()} windows")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.float32), torch.tensor(
            self.labels[idx], dtype=torch.long
        )


class BinaryAnxietyClassifier(nn.Module):
    def __init__(self):
        super(BinaryAnxietyClassifier, self).__init__()

        # load frozen SingLEM feature extractor
        config = Config()
        config.mask_prob = 0.0
        self.feature_extractor = EEGEncoder(config)
        self.feature_extractor.load_state_dict(
            torch.load(PRETRAINED_WEIGHTS_PATH, map_location="cpu")
        )

        # unfreeze the foundation model so it can adapt to fine-tuning
        for param in self.feature_extractor.parameters():
            param.requires_grad = True

        # custom classification head (num_classes = 2)
        self.classifier = nn.Sequential(
            nn.Linear(5 * 16, 64),  # 5 tokens * 16 hidden_dim from the encoder
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        features, _, _ = self.feature_extractor(x)
        features = features.view(features.size(0), -1)  # flatten token dimensions
        return self.classifier(features)


def train_model(participant_regex=None):
    train_dataset = BinaryEEGDataset(mode="train", participant_regex=participant_regex)
    val_dataset = BinaryEEGDataset(mode="val", participant_regex=participant_regex)

    if len(train_dataset) == 0:
        print("no valid EEG data found matching criteria; exiting")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\ntraining on device: {device}")

    model = BinaryAnxietyClassifier().to(device)
    criterion = nn.CrossEntropyLoss()

    # differential learning rates: train backbone slowly and train head fast
    optimizer = optim.Adam(
        [
            {"params": model.feature_extractor.parameters(), "lr": 1e-5},
            {"params": model.classifier.parameters(), "lr": 1e-3},
        ]
    )

    best_val_acc = 0.0

    print("beginning training loop...")
    for epoch in range(EPOCHS):
        # training phase
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        train_acc = (train_correct / train_total) * 100
        avg_train_loss = train_loss / len(train_loader)

        # validation phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        val_acc = (val_correct / val_total) * 100
        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch [{epoch + 1:03d}/{EPOCHS}] | "
            f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.1f}% | "
            f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.1f}%"
        )

        # save the model ONLY if it improves on the true validation set
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "singlem_binary_head.pth")

    print(f"\ntraining complete. best validation accuracy: {best_val_acc:.1f}%")
    print("best weights saved to 'singlem_binary_head.pth'")


if __name__ == "__main__":
    train_model(participant_regex=TARGET_PARTICIPANT_REGEX)
