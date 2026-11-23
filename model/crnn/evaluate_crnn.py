import os
import sys
import time
import unicodedata
import json
import cv2
import numpy as np
import pandas as pd
import torch
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

@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CRNN Inference Device: {device}")

    model_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(model_dir, "crnn_best.pth")
    vocab_path = os.path.join(model_dir, "vocab.json")
    test_parquet = os.path.normpath(os.path.join(model_dir, "..", "..", "dataset", "data", "test.parquet"))

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    vocab = Vocab.load(vocab_path)
    ckpt = torch.load(model_path, map_location=device)
    model = CRNN(num_classes=len(vocab), in_channels=1, hidden_size=256).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded CRNN checkpoint from epoch {ckpt['epoch']}")

    df_test = pd.read_parquet(test_parquet)
    print(f"Evaluating on all {len(df_test)} test samples...")

    records = []
    latencies = []

    for i in range(len(df_test)):
        row = df_test.iloc[i]
        raw = row.get("image.bytes", row.get("image_bytes"))
        nparr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        t0 = time.time()
        norm_img, _ = preprocess_image(img, TARGET_HEIGHT, MAX_WIDTH)
        x = torch.from_numpy(norm_img).unsqueeze(0).unsqueeze(0).float().to(device)

        logits = model(x) # (1, T, C)
        preds = torch.argmax(logits, dim=2)[0].cpu().numpy().tolist()
        pred_text = vocab.decode_ctc(preds)
        latency = (time.time() - t0) * 1000.0
        latencies.append(latency)

        gt_text = unicodedata.normalize("NFC", str(row["label"]).strip())
        cer = compute_cer(gt_text, pred_text)
        acc = max(0.0, 1.0 - cer) * 100.0
        exact = 1.0 if gt_text == pred_text else 0.0

        records.append({
            "filename": row.get("filename", f"sample_{i}.png"),
            "ground_truth": gt_text,
            "prediction": pred_text,
            "char_accuracy": acc,
            "cer": cer,
            "exact_match": exact,
            "latency_ms": latency,
        })

    eval_df = pd.DataFrame(records)
    print("\n" + "=" * 60)
    print("           CRNN QUANTITATIVE EVALUATION ON TEST SET")
    print("=" * 60)
    print(f"  Total Test Samples:      {len(eval_df)}")
    print(f"  Mean Character Accuracy: {eval_df['char_accuracy'].mean():.2f}%")
    print(f"  Mean CER:                {eval_df['cer'].mean():.3f}")
    print(f"  Exact Word Match:        {eval_df['exact_match'].mean() * 100.0:.2f}%")
    print(f"  Average Latency:         {np.mean(latencies):.2f} ms / word")
    print("=" * 60)

    print("\n[FIRST 10 TEST SAMPLES COMPARISON]:")
    for i in range(min(10, len(eval_df))):
        r = eval_df.iloc[i]
        match_sym = "✓" if r["exact_match"] == 1.0 else "✗"
        print(f"#{i:02d} [{match_sym}] GT:   '{r['ground_truth']}'")
        print(f"    PRED: '{r['prediction']}' (Acc: {r['char_accuracy']:.1f}%, CER: {r['cer']:.2f}, Latency: {r['latency_ms']:.1f}ms)")

if __name__ == "__main__":
    main()
