import os
import re
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import signal
from torch.utils.data import DataLoader, Dataset

# import SingLEM from git submodule
sys.path.append(os.path.join(os.path.dirname(__file__), "SingLEM"))
from SingLEM.model import Config, EEGEncoder

# configuration
DATA_BASE_DIR = "data"
PRETRAINED_WEIGHTS_PATH = "SingLEM/weights/singlem_pretrained.pt"
EEG_COLUMN = "eegRawValueVolts"

BATCH_SIZE = 32
EPOCHS = 50
DISCARD_SECONDS = 60
PATIENCE = 15
POOR_SIGNAL = 45
UNFREEZE_LAST_N_PARAMS = 36
TARGET_PARTICIPANT_REGEX = r"^Andy$"


def preprocess_eeg(raw_volts):
    """bandpass (0.5-50hz), notch 60hz, downsample 512->128, scale."""
    b, a = signal.butter(4, [0.5, 50.0], btype="bandpass", fs=512)
    filtered = signal.filtfilt(b, a, raw_volts)
    b_notch, a_notch = signal.iirnotch(60.0, 30.0, fs=512)
    filtered = signal.filtfilt(b_notch, a_notch, filtered)
    resampled = signal.decimate(filtered, q=4)
    return resampled * 1e4


class BinaryEEGDataset(Dataset):
    def __init__(self, mode="train", participant_regex=None, seed=42):
        self.samples = []
        self.labels = []
        self.mode = mode

        level_mapping = {1: 0, 2: 1}
        regex = re.compile(participant_regex) if participant_regex else None
        class_files = {0: [], 1: []}

        print(f"loading silent EEG data for {mode.lower()}...")

        for level, cls in level_mapping.items():
            folder = os.path.join(DATA_BASE_DIR, f"level_{level}")
            if not os.path.isdir(folder):
                continue

            for fname in os.listdir(folder):
                if not fname.endswith(".csv"):
                    continue
                parts = fname[:-4].split("_")
                if len(parts) < 6:
                    continue
                participant_name, datatype, test_activity, test_condition = (
                    parts[2],
                    parts[3],
                    parts[4],
                    parts[5],
                )
                if datatype != "EEG" or test_condition != "Silent":
                    continue
                if regex and not regex.search(participant_name):
                    continue
                # class-specific task filters
                if cls == 0 and test_activity != "Reading":
                    continue
                if cls == 1 and test_activity not in ("Stroop", "Typing"):
                    continue
                class_files[cls].append((os.path.join(folder, fname), cls, fname))

        # stratified split by class
        np.random.seed(seed)
        for cls in class_files:
            np.random.shuffle(class_files[cls])

        split_files = []
        for cls, files in class_files.items():
            sidx = int(len(files) * 0.8)
            if mode == "train":
                split_files += files[:sidx]
            else:
                split_files += files[sidx:]

        # load files and extract cleaned windows
        window_size_128 = 5 * 128
        step_size_128 = 128
        discard_samples = DISCARD_SECONDS * 512

        for path, cls, fname in split_files:
            df = pd.read_csv(path)
            if EEG_COLUMN not in df.columns or "poorSignal" not in df.columns:
                continue
            raw = df[EEG_COLUMN].values
            poor = df["poorSignal"].values
            if len(raw) <= discard_samples:
                if mode == "train":
                    print(f"warning: {fname} shorter than {DISCARD_SECONDS}s; skipping")
                continue
            raw = raw[discard_samples:]
            poor = poor[discard_samples:]
            processed = preprocess_eeg(raw)

            is_clean = poor < POOR_SIGNAL
            # extract contiguous clean blocks
            idx = 0
            n = len(is_clean)
            while idx < n:
                if not is_clean[idx]:
                    idx += 1
                    continue
                start = idx
                while idx < n and is_clean[idx]:
                    idx += 1
                end = idx
                start_128 = start // 4
                end_128 = end // 4
                if end_128 - start_128 >= window_size_128:
                    for ws in range(
                        start_128, end_128 - window_size_128 + 1, step_size_128
                    ):
                        w = processed[ws : ws + window_size_128]
                        self.samples.append(w.reshape(5, 128))
                        self.labels.append(cls)

        print(
            f"loaded {len(self.samples)} clean {mode.lower()} windows from {len(split_files)} files"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.samples[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


class BinaryAnxietyClassifier(nn.Module):
    def __init__(self, unfreeze_last_n=0):
        super().__init__()
        config = Config()
        config.mask_prob = 0.0
        self.feature_extractor = EEGEncoder(config)
        self.feature_extractor.load_state_dict(
            torch.load(PRETRAINED_WEIGHTS_PATH, map_location="cpu")
        )
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        if unfreeze_last_n > 0:
            params = list(self.feature_extractor.parameters())
            for p in params[-unfreeze_last_n:]:
                p.requires_grad = True
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

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("no valid train/val data found; exiting")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\ntraining on device: {device}")

    model = BinaryAnxietyClassifier(unfreeze_last_n=UNFREEZE_LAST_N_PARAMS).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer_params = [{"params": model.classifier.parameters(), "lr": 1e-3}]
    backbone_params = [
        p for p in model.feature_extractor.parameters() if p.requires_grad
    ]
    if backbone_params:
        print(
            f"info: partial fine-tuning enabled. {len(backbone_params)} backbone tensors will train with a micro-learning rate."
        )
        optimizer_params.append({"params": backbone_params, "lr": 1e-5})

    optimizer = optim.AdamW(optimizer_params, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    print("beginning training loop...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        # keep backbone in eval() to preserve pretrained statistics
        model.feature_extractor.eval()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            train_total += batch_y.size(0)
            train_correct += (preds == batch_y).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100.0 * train_correct / train_total

        # validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                val_total += batch_y.size(0)
                val_correct += (preds == batch_y).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * val_correct / val_total

        print(
            f"epoch [{epoch:02d}/{EPOCHS}] | train loss: {avg_train_loss:.4f}, acc: {train_acc:.1f}% | val loss: {avg_val_loss:.4f}, acc: {val_acc:.1f}%"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), "singlem_binary_head.pth")
        else:
            patience_counter += 1
            print(
                f"no improvement in val loss. early stopping counter: {patience_counter}/{PATIENCE}"
            )
            if patience_counter >= PATIENCE:
                print(f"\nearly stopping triggered at epoch {epoch}.")
                break

        scheduler.step()

    print("\ntraining complete")
    print(f"optimal weights found at epoch {best_epoch}")
    print(f"best validation loss: {best_val_loss:.4f}")
    print(f"corresponding validation accuracy: {best_val_acc:.1f}%")
    print("model saved to 'singlem_binary_head.pth'")


if __name__ == "__main__":
    train_model(participant_regex=TARGET_PARTICIPANT_REGEX)

