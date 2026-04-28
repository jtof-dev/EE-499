import os
import random
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
PATIENCE = 15  # Early stopping patience
POOR_SIGNAL = 45

# --- NEW CONFIG ---
# Number of parameters at the end of the foundation model to unfreeze.
# 0 = completely frozen (Linear Probing)
# 12-24 = roughly 1 to 2 transformer blocks depending on SingLEM's architecture (Partial Fine-Tuning)
UNFREEZE_LAST_N_PARAMS = 36

# set to None to include all participants, or use a regex string like r"P0[1-5]"
TARGET_PARTICIPANT_REGEX = r"^Andy$"


def preprocess_eeg(raw_volts):
    """
    SingLEM 128Hz preprocessing pipeline.
    Applies a bandpass filter (0.5-50Hz) to isolate brainwaves, a notch filter (60Hz)
    to remove electrical line noise, downsamples to 128Hz, and scales the signal.
    """
    # bandpass filter (0.5Hz - 50Hz)
    b, a = signal.butter(4, [0.5, 50.0], btype="bandpass", fs=512)
    filtered = signal.filtfilt(b, a, raw_volts)

    # notch filter (60Hz)
    b_notch, a_notch = signal.iirnotch(60.0, 30.0, fs=512)
    filtered = signal.filtfilt(b_notch, a_notch, filtered)

    # downsample (512Hz -> 128Hz); q=4 downsampling factor (512 / 4 = 128)
    resampled = signal.decimate(filtered, q=4)
    scaled = resampled * 1e4  # scale to [-1, 1] typical range for SingLEM

    # NOTE: Z-score normalization removed to preserve absolute amplitude required by SingLEM
    return scaled


class BinaryEEGDataset(Dataset):
    def __init__(self, mode="train", participant_regex=None, seed=42):
        self.samples = []
        self.labels = []
        self.mode = mode

        level_mapping = {1: 0, 2: 1}
        print(f"Loading Silent EEG data for {mode.upper()}...")

        regex_pattern = re.compile(participant_regex) if participant_regex else None

        # 1. Gather valid files, separated by class to ensure balanced splits
        class_0_files = []
        class_1_files = []

        for level, py_class in level_mapping.items():
            folder = os.path.join(DATA_BASE_DIR, f"level_{level}")
            if not os.path.exists(folder):
                continue

            for file in os.listdir(folder):
                if not file.endswith(".csv"):
                    continue

                parts = file.replace(".csv", "").split("_")
                if len(parts) >= 6:
                    participant_name = parts[2]
                    datatype = parts[3]
                    test_activity = parts[
                        4
                    ]  # NEW: Extract the specific task (Reading, Stroop, etc.)
                    test_condition = parts[5]
                else:
                    continue

                # 1. Base environmental filters (must be EEG data, and must be in the Silent control group)
                if datatype != "EEG" or test_condition != "Silent":
                    continue

                # 2. Participant filter
                if regex_pattern and not regex_pattern.search(participant_name):
                    continue

                # 3. STRICT COGNITIVE ISOLATION FILTER
                # Class 0 (Level 1): Only allow the passive 'Reading' task
                if py_class == 0 and test_activity != "Reading":
                    continue

                # Class 1 (Level 2): Only allow the high-stress 'Stroop' task (excludes Typing)
                # if py_class == 1 and test_activity != "Stroop":
                if py_class == 1 and test_activity not in ["Stroop", "Typing"]:
                    continue
                path = os.path.join(folder, file)

                # Sort files into their respective class buckets
                if py_class == 0:
                    class_0_files.append((path, py_class, file))
                else:
                    class_1_files.append((path, py_class, file))

        # 2. Shuffle and Stratify the split
        random.seed(seed)
        random.shuffle(class_0_files)
        random.shuffle(class_1_files)

        split_idx_0 = int(len(class_0_files) * 0.8)
        split_idx_1 = int(len(class_1_files) * 0.8)

        if mode == "train":
            target_files = class_0_files[:split_idx_0] + class_1_files[:split_idx_1]
        elif mode == "val":
            target_files = class_0_files[split_idx_0:] + class_1_files[split_idx_1:]

        # 3. Process only the assigned files for this split
        for path, py_class, file_name in target_files:
            df = pd.read_csv(path)

            if EEG_COLUMN not in df.columns or "poorSignal" not in df.columns:
                continue

            raw_volts = df[EEG_COLUMN].values
            poor_signal = df["poorSignal"].values

            samples_to_discard = DISCARD_SECONDS * 512

            if len(raw_volts) <= samples_to_discard:
                if mode == "train":
                    print(
                        f"warning: {file_name} is shorter than {DISCARD_SECONDS}s; skipping"
                    )
                continue

            raw_volts = raw_volts[samples_to_discard:]
            poor_signal = poor_signal[samples_to_discard:]

            processed_eeg = preprocess_eeg(raw_volts)

            window_size_128 = 5 * 128
            step_size_128 = 128

            is_clean = poor_signal < POOR_SIGNAL

            clean_blocks = []
            current_start = None

            for i, clean in enumerate(is_clean):
                if clean and current_start is None:
                    current_start = i
                elif not clean and current_start is not None:
                    clean_blocks.append((current_start, i))
                    current_start = None

            if current_start is not None:
                clean_blocks.append((current_start, len(is_clean)))

            for start_512, end_512 in clean_blocks:
                start_128 = start_512 // 4
                end_128 = end_512 // 4

                if (end_128 - start_128) >= window_size_128:
                    for window_start in range(
                        start_128, end_128 - window_size_128 + 1, step_size_128
                    ):
                        window = processed_eeg[
                            window_start : window_start + window_size_128
                        ]
                        reshaped = window.reshape(5, 128)

                        self.samples.append(reshaped)
                        self.labels.append(py_class)

        print(
            f"loaded {len(self.samples)} clean {mode.upper()} windows from {len(target_files)} files"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.float32), torch.tensor(
            self.labels[idx], dtype=torch.long
        )


class BinaryAnxietyClassifier(nn.Module):
    def __init__(self, unfreeze_last_n=0):
        super(BinaryAnxietyClassifier, self).__init__()

        # load frozen SingLEM feature extractor
        config = Config()
        config.mask_prob = 0.0
        self.feature_extractor = EEGEncoder(config)
        self.feature_extractor.load_state_dict(
            torch.load(PRETRAINED_WEIGHTS_PATH, map_location="cpu")
        )

        # 1. Freeze everything by default
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # 2. Selectively unfreeze the last N parameters if requested
        if unfreeze_last_n > 0:
            params = list(self.feature_extractor.parameters())
            for param in params[-unfreeze_last_n:]:
                param.requires_grad = True

        # custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(5 * 16, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        features, _, _ = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
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

    model = BinaryAnxietyClassifier(unfreeze_last_n=UNFREEZE_LAST_N_PARAMS).to(device)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Differential Learning Rates Setup
    optimizer_params = [{"params": model.classifier.parameters(), "lr": 1e-3}]

    # Grab any backbone parameters that we dynamically unfreezed
    backbone_params = [
        p for p in model.feature_extractor.parameters() if p.requires_grad
    ]
    if backbone_params:
        print(
            f"info: partial fine-tuning enabled. {len(backbone_params)} backbone tensors will train with a micro-learning rate."
        )
        optimizer_params.append(
            {"params": backbone_params, "lr": 1e-5}
        )  # Micro LR to protect pretrained weights

    # optimizer = optim.Adam(optimizer_params)

    optimizer = optim.AdamW(optimizer_params, weight_decay=0.01)

    # Add the scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Enhanced Logging Trackers
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    print("beginning training loop...")
    for epoch in range(EPOCHS):
        # training phase
        model.train()

        # Keep the backbone in eval mode even if partially unfreezed to prevent
        # BatchNorm/Dropout shifting which ruins pre-trained stability.
        model.feature_extractor.eval()

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
            f"Epoch [{epoch + 1:02d}/{EPOCHS}] | "
            f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.1f}% | "
            f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.1f}%"
        )

        # early stopping and model saving logic based on loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), "singlem_binary_head.pth")
        else:
            patience_counter += 1
            print(
                f"-> no improvement in val loss. early stopping counter: {patience_counter}/{PATIENCE}"
            )

            if patience_counter >= PATIENCE:
                print(f"\n[!] early stopping triggered at epoch {epoch + 1}.")
                print("validation loss has stopped improving. preventing overfitting.")
                break

        # Step the scheduler at the VERY END of the epoch (outside the batch loop)
    scheduler.step()

    print("\n--- TRAINING COMPLETE ---")
    print(f"Optimal weights found at Epoch {best_epoch}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Corresponding Validation Accuracy: {best_val_acc:.1f}%")
    print("Model saved to 'singlem_binary_head.pth'")


if __name__ == "__main__":
    train_model(participant_regex=TARGET_PARTICIPANT_REGEX)
