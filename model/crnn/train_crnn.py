import os
import sys
import time
import unicodedata
import json
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from crnn_model import CRNN, Vocab, preprocess_image, TARGET_HEIGHT, MAX_WIDTH

def levenshtein(s1, s2):
    s1, s2 = unicodedata.normalize("NFC", s1.strip()), unicodedata.normalize("NFC", s2.strip())
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[m][n]

def compute_cer(ref, hyp):
    ref_norm = unicodedata.normalize("NFC", ref.strip())
    if not ref_norm: return 0.0 if not hyp else 1.0
    return levenshtein(ref_norm, hyp) / float(len(ref_norm))

class ParquetOCRDataset(Dataset):
    def __init__(self, parquet_path, vocab, is_train=True):
        self.df = pd.read_parquet(parquet_path)
        self.vocab = vocab
        self.is_train = is_train

        # Filter out empty or whitespace-only labels
        valid_indices = []
        for i in range(len(self.df)):
            lbl = str(self.df.iloc[i]["label"]).strip()
            if len(lbl) > 0 and any(c in self.vocab.char2idx for c in lbl):
                valid_indices.append(i)
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)

        # Pre-cache preprocessed images and tensors in RAM for blazing fast training
        print(f"Caching {len(self.df)} {'train' if is_train else 'test'} samples into memory...")
        self.cached_samples = []
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            raw = row.get("image.bytes", row.get("image_bytes"))
            nparr = np.frombuffer(raw, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            label_text = unicodedata.normalize("NFC", str(row["label"]).strip())
            target_indices = self.vocab.encode(label_text)

            norm_img, valid_w = preprocess_image(img, TARGET_HEIGHT, MAX_WIDTH)
            img_tensor = torch.from_numpy(norm_img).unsqueeze(0).float()
            time_steps = max(1, valid_w // 4)

            self.cached_samples.append({
                "img": img_tensor,
                "target": torch.tensor(target_indices, dtype=torch.long),
                "label_text": label_text,
                "valid_w": valid_w,
                "time_steps": time_steps,
            })

    def __len__(self):
        return len(self.cached_samples)

    def __getitem__(self, idx):
        return self.cached_samples[idx]

def collate_fn(batch):
    batch_size = len(batch)
    imgs = torch.stack([item["img"] for item in batch], dim=0)

    # Flatten targets for CTCLoss
    targets = []
    target_lengths = []
    input_lengths = []
    labels = []

    for item in batch:
        t = item["target"]
        targets.extend(t.tolist())
        target_lengths.append(len(t))
        # Ensure input length is at least target length for CTC
        input_len = max(len(t), item["time_steps"])
        input_lengths.append(input_len)
        labels.append(item["label_text"])

    targets = torch.tensor(targets, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)

    return {
        "imgs": imgs,
        "targets": targets,
        "target_lengths": target_lengths,
        "input_lengths": input_lengths,
        "labels": labels,
    }

def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        imgs = batch["imgs"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
            logits = model(imgs)  # (B, T, C)
            log_probs = logits.log_softmax(2).permute(1, 0, 2)  # (T, B, C) for CTCLoss
            input_lengths = torch.full((imgs.size(0),), logits.size(1), dtype=torch.long, device=device)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)

        if not torch.isfinite(loss):
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(1, total_samples)

@torch.no_grad()
def evaluate(model, dataloader, vocab, device):
    model.eval()
    cer_list = []
    exact_matches = 0
    total_samples = 0
    predictions = []

    for batch in dataloader:
        imgs = batch["imgs"].to(device)
        labels = batch["labels"]

        logits = model(imgs)  # (B, T, C)
        preds = torch.argmax(logits, dim=2)  # (B, T)

        for i in range(len(labels)):
            pred_indices = preds[i].cpu().numpy().tolist()
            pred_text = vocab.decode_ctc(pred_indices)
            gt_text = labels[i]

            cer = compute_cer(gt_text, pred_text)
            cer_list.append(cer)
            if unicodedata.normalize("NFC", gt_text) == pred_text:
                exact_matches += 1
            total_samples += 1

            if len(predictions) < 5:
                predictions.append((gt_text, pred_text, cer))

    mean_cer = float(np.mean(cer_list)) if cer_list else 1.0
    char_acc = max(0.0, 1.0 - mean_cer) * 100.0
    word_acc = (exact_matches / max(1, total_samples)) * 100.0
    return {
        "mean_cer": mean_cer,
        "char_acc": char_acc,
        "word_acc": word_acc,
        "samples": predictions,
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using compute device:", device)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(script_dir, "..", ".."))

    train_path = os.path.join(project_root, "dataset", "data", "train.parquet")
    test_path = os.path.join(project_root, "dataset", "data", "test.parquet")
    out_dir = script_dir
    os.makedirs(out_dir, exist_ok=True)
    vocab_path = os.path.join(out_dir, "vocab.json")
    best_model_path = os.path.join(out_dir, "crnn_best.pth")

    # Build or load vocabulary
    print("Building vocabulary from dataset labels...")
    df_tr = pd.read_parquet(train_path)
    df_te = pd.read_parquet(test_path)
    chars = set()
    for lbl in list(df_tr["label"]) + list(df_te["label"]):
        norm = unicodedata.normalize("NFC", str(lbl).strip())
        chars.update(norm)

    vocab = Vocab(chars)
    vocab.save(vocab_path)
    print(f"Vocabulary size: {len(vocab)} (including CTC blank token)")

    # Data loaders
    train_dataset = ParquetOCRDataset(train_path, vocab, is_train=True)
    test_dataset = ParquetOCRDataset(test_path, vocab, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # Initialize CRNN Model
    model = CRNN(num_classes=len(vocab), in_channels=1, hidden_size=256).to(device)
    criterion = nn.CTCLoss(blank=vocab.blank_idx, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    epochs = 35
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_char_acc = 0.0
    best_cer = 999.0

    print("\n" + "=" * 65)
    print("              STARTING CRNN SANSKRIT OCR TRAINING")
    print("=" * 65)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        scheduler.step()
        elapsed = time.time() - t0

        # Evaluate on test set
        metrics = evaluate(model, test_loader, vocab, device)
        char_acc = metrics["char_acc"]
        mean_cer = metrics["mean_cer"]
        word_acc = metrics["word_acc"]

        is_best = mean_cer < best_cer
        if is_best:
            best_cer = mean_cer
            best_char_acc = char_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "vocab_size": len(vocab),
                "best_char_acc": best_char_acc,
                "best_cer": best_cer,
                "best_word_acc": word_acc,
            }, best_model_path)
            saved_tag = " ⭐ (Best saved)"
        else:
            saved_tag = ""

        print(
            f"Epoch {epoch:02d}/{epochs} [{elapsed:.1f}s] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test CER: {mean_cer:.3f} | "
            f"Test Char Acc: {char_acc:.1f}% | "
            f"Word Match: {word_acc:.1f}%{saved_tag}"
        )

        if epoch % 5 == 0 or is_best:
            print("  [Sample Predictions]:")
            for gt, pred, cer in metrics["samples"][:3]:
                print(f"    GT:   '{gt}'\n    PRED: '{pred}' (CER: {cer:.2f})")

    print("\n" + "=" * 65)
    print("                      TRAINING COMPLETE")
    print(f"  Best Test CER:           {best_cer:.3f}")
    print(f"  Best Character Accuracy: {best_char_acc:.1f}%")
    print(f"  Best Model Saved To:     {best_model_path}")
    print("=" * 65)

if __name__ == "__main__":
    main()
