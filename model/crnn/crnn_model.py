import os
import json
import unicodedata
import torch
import torch.nn as nn
import numpy as np
import cv2

TARGET_HEIGHT = 32
MAX_WIDTH = 448

class CRNN(nn.Module):
    """
    Standard CRNN architecture for Text Recognition (Shi et al., PAMI 2017):
    CNN (VGG-style feature extractor with asymmetric pooling to preserve width)
    + 2-layer Bidirectional LSTM
    + CTC transcription layer.
    """
    def __init__(self, num_classes, in_channels=1, hidden_size=256):
        super().__init__()
        self.num_classes = num_classes

        # CNN Backbone: maps (B, 1, 32, W) -> (B, 512, 1, W/4)
        self.cnn = nn.Sequential(
            # Stage 1: (32, W) -> (16, W/2)
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            # Stage 2: (16, W/2) -> (8, W/4)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            # Stage 3: (8, W/4) -> (4, W/4)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),

            # Stage 4: (4, W/4) -> (2, W/4)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),

            # Stage 5: (2, W/4) -> (1, W/4)
            nn.Conv2d(512, 512, kernel_size=(2, 1), stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )

        # RNN: 2-layer Bidirectional LSTM
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2,
        )

        # Fully-connected projection: (B, T, hidden*2) -> (B, T, num_classes)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x: (B, 1, H, W)
        conv = self.cnn(x)  # (B, 512, 1, T)
        b, c, h, t = conv.size()
        assert h == 1, f"Expected height 1 after CNN, got {h}"

        # Squeeze height and permute to (B, T, C)
        conv = conv.squeeze(2).permute(0, 2, 1)

        # RNN forward
        recurrent, _ = self.rnn(conv)  # (B, T, 2*hidden)

        # Classifier logits
        logits = self.fc(recurrent)  # (B, T, num_classes)
        return logits


class Vocab:
    """Manages character to index and index to character mapping."""
    def __init__(self, chars=None):
        # 0 is reserved for CTC blank token
        self.blank_idx = 0
        self.pad_idx = 0
        if chars is None:
            self.chars = []
            self.char2idx = {}
            self.idx2char = {}
        else:
            self.chars = sorted(list(chars))
            self.char2idx = {c: i + 1 for i, c in enumerate(self.chars)}
            self.idx2char = {i + 1: c for i, c in enumerate(self.chars)}
            self.idx2char[self.blank_idx] = ""

    def __len__(self):
        return len(self.chars) + 1  # +1 for blank token

    def encode(self, text):
        norm = unicodedata.normalize("NFC", str(text).strip())
        indices = []
        for c in norm:
            if c in self.char2idx:
                indices.append(self.char2idx[c])
        return indices

    def decode(self, indices):
        chars = []
        for idx in indices:
            if idx != self.blank_idx and idx in self.idx2char:
                chars.append(self.idx2char[idx])
        return unicodedata.normalize("NFC", "".join(chars))

    def decode_ctc(self, sequence):
        """Greedy CTC decoding (collapse consecutive duplicates, remove blanks)."""
        collapsed = []
        prev = self.blank_idx
        for idx in sequence:
            if idx != prev:
                if idx != self.blank_idx:
                    collapsed.append(idx)
            prev = idx
        return self.decode(collapsed)

    def save(self, path):
        data = {
            "chars": self.chars,
            "char2idx": self.char2idx,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls()
        vocab.chars = data["chars"]
        vocab.char2idx = {k: int(v) for k, v in data["char2idx"].items()}
        vocab.idx2char = {int(v): k for k, v in data["char2idx"].items()}
        vocab.idx2char[vocab.blank_idx] = ""
        return vocab


def preprocess_image(img, target_h=TARGET_HEIGHT, max_w=MAX_WIDTH):
    """
    Standardizes image:
    1. Converts to grayscale.
    2. Resizes height to target_h while maintaining aspect ratio.
    3. Pads width with white (255) up to max_w.
    4. Normalizes to [-1.0, 1.0].
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    h, w = gray.shape
    scale = target_h / float(max(1, h))
    new_w = min(max_w, max(1, round(w * scale)))
    resized = cv2.resize(gray, (new_w, target_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)

    canvas = np.full((target_h, max_w), 255, dtype=np.uint8)
    canvas[:, :new_w] = resized

    # Normalize: ink (dark) -> +1.0, background (white) -> -1.0
    norm = (1.0 - canvas.astype(np.float32) / 127.5)  # white (255) -> -1.0, black (0) -> +1.0
    return norm, new_w
