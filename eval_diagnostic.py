import os
import sys
import time
import unicodedata
import cv2
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

class CNN3Stage(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.10),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.20),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.30),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def load_model_and_classes(folder_name, weight_file):
    base = os.path.join("model", folder_name)
    ckpt_path = os.path.join(base, weight_file)
    class_file = os.path.join(base, "classes.json")
    
    with open(class_file, encoding="utf-8") as f:
        class_meta = json.load(f)

    char_map = {}
    for item in class_meta:
        glyph = item.get("sign") or item.get("char") or ""
        char_map[item["name"]] = glyph
        if "folder" in item:
            char_map[item["folder"]] = glyph

    ckpt = torch.load(ckpt_path, map_location=device)
    classes = ckpt["classes"]
    model = CNN3Stage(ckpt["num_classes"]).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    return model, classes, char_map

full_model, full_classes, full_map = load_model_and_classes("full_character", "full_character.pth")
half_model, half_classes, half_map = load_model_and_classes("half_character", "half_character.pth")
comp_model, comp_classes, comp_map = load_model_and_classes("compound_character", "compound_character.pth")
lower_model, lower_classes, lower_map = load_model_and_classes("lower_modifier", "lower_modifier.pth")
upper_model, upper_classes, upper_map = load_model_and_classes("upper_modifier", "upper_modifier.pth")

REF_HEIGHT = 34
CANVAS_SIZE = 64

def normalize_crop_to_64(gray_crop, base_height=REF_HEIGHT):
    scale = REF_HEIGHT / float(max(1, base_height))
    h, w = gray_crop.shape
    new_h = max(1, round(h * scale))
    new_w = max(1, round(w * scale))

    scaled = cv2.resize(gray_crop, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    if max(new_h, new_w) > CANVAS_SIZE - 4:
        shrink = (CANVAS_SIZE - 4) / max(new_h, new_w)
        scaled = cv2.resize(scaled, (max(1, round(new_w * shrink)), max(1, round(new_h * shrink))), interpolation=cv2.INTER_AREA)

    canvas = np.full((CANVAS_SIZE, CANVAS_SIZE), 255, dtype=np.uint8)
    y_off = (CANVAS_SIZE - scaled.shape[0]) // 2
    x_off = (CANVAS_SIZE - scaled.shape[1]) // 2
    canvas[y_off : y_off + scaled.shape[0], x_off : x_off + scaled.shape[1]] = scaled

    tensor = (1.0 - torch.from_numpy(canvas).float() / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    return tensor

def segment_word_image(input_img):
    if len(input_img.shape) == 3:
        gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = input_img.copy()

    bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)
    if np.sum(bin_img == 0) > np.sum(bin_img == 255):
        bin_img = cv2.bitwise_not(bin_img)
    padded = cv2.copyMakeBorder(bin_img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)

    binary_sh = (padded == 0).astype("uint8")
    h_profile = np.sum(binary_sh, axis=1)
    max_h = np.max(h_profile)
    if max_h == 0:
        return {"upper": [], "middle": [], "lower": [], "base_height": 28, "vis": padded}

    threshold = 0.65 * max_h
    rows = np.where(h_profile >= threshold)[0]
    valid_bands = []
    if len(rows) > 0:
        bands = [[rows[0]]]
        for r in rows[1:]:
            if r == bands[-1][-1] + 1:
                bands[-1].append(r)
            else:
                bands.append([r])
        w = padded.shape[1]
        valid_bands = [b for b in bands if np.mean([np.sum(padded[r] == 0) / w for r in b]) > 0.35]

    sh_removed = padded.copy()
    for b in valid_bands:
        mid = int(np.mean(b))
        for t in range(-1, 2):
            if 0 <= mid + t < sh_removed.shape[0]:
                sh_removed[mid + t, :] = 255

    reconnected = sh_removed.copy()
    if valid_bands:
        cut_rows = [int(np.mean(b)) for b in valid_bands]
        z_top, z_bottom = max(0, min(cut_rows) - 3), min(padded.shape[0], max(cut_rows) + 3)
        v_prof = np.sum((sh_removed == 0).astype("uint8"), axis=0)
        max_vp = np.max(v_prof)
        if max_vp > 0:
            for col in np.where(v_prof >= 0.70 * max_vp)[0]:
                r = z_top
                while r < z_bottom:
                    if reconnected[r, col] == 0:
                        r2 = r + 1
                        gap = 0
                        while r2 < z_bottom and reconnected[r2, col] == 255:
                            gap += 1
                            if gap > 5: break
                            r2 += 1
                        if r2 < z_bottom and 0 < gap <= 5:
                            reconnected[r + 1 : r2, col] = 0
                            r = r2
                        else: r += 1
                    else: r += 1

    binary = (reconnected == 0).astype("uint8")
    num_labels_init, _, stats_init, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels_init <= 1:
        return {"upper": [], "middle": [], "lower": [], "base_height": 28, "vis": reconnected}

    heights = stats_init[1:, cv2.CC_STAT_HEIGHT]
    centers_y = stats_init[1:, cv2.CC_STAT_TOP] + (heights / 2.0)
    med_h = float(np.median(heights))
    med_c = float(np.median(centers_y))
    mid_top = int(med_c - 0.55 * med_h)
    mid_bottom = int(med_c + 0.55 * med_h)

    split_binary = binary.copy()
    margin = max(2, int(med_h * 0.20))
    for p in range(1, num_labels_init):
        x, y, w_comp, h = stats_init[p, :4]
        if y < mid_bottom and (y + h) > (mid_bottom + margin):
            roi = split_binary[y : y + h, x : x + w_comp]
            loc_mid = mid_bottom - y
            s_start, s_end = max(0, loc_mid - margin), min(roi.shape[0], loc_mid + margin)
            if s_start < s_end:
                r_sums = np.sum(roi[s_start:s_end, :], axis=1)
                if len(r_sums) > 0:
                    split_y = y + s_start + int(np.argmin(r_sums))
                    split_binary[max(0, split_y - 1) : split_y, x : x + w_comp] = 0

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(split_binary, connectivity=8)
    categorized = {"upper": [], "middle": [], "lower": []}

    for p in range(1, num_labels):
        x, y, w_comp, h, area = stats[p]
        if area < 5: continue
        cy = y + (h / 2.0)
        zone = "upper" if cy < mid_top else ("lower" if cy > mid_bottom else "middle")

        crop = reconnected[max(0, y - 1) : y + h + 1, max(0, x - 1) : x + w_comp + 1]
        categorized[zone].append({
            "x": int(x), "y": int(y), "w": int(w_comp), "h": int(h),
            "cx": float(centroids[p][0]), "crop": crop,
            "aspect_ratio": float(w_comp) / max(1.0, float(h))
        })

    for z in categorized:
        categorized[z].sort(key=lambda item: item["x"])

    m_heights = [c["h"] for c in categorized["middle"]]
    base_h = float(np.median(m_heights)) if m_heights else med_h
    return {"upper": categorized["upper"], "middle": categorized["middle"], "lower": categorized["lower"], "base_height": base_h, "vis": reconnected}

@torch.no_grad()
def predict_glyph(model, classes, char_map, crop, base_height):
    x = normalize_crop_to_64(crop, base_height)
    probs = torch.softmax(model(x), dim=1)[0]
    idx = torch.argmax(probs).item()
    cls_name = classes[idx]
    return {"class": cls_name, "text": char_map.get(cls_name, ""), "conf": probs[idx].item()}

@torch.no_grad()
def route_middle_component(comp, base_height, threshold=0.82):
    crop = comp["crop"]
    w, h = comp["w"], comp["h"]

    if w <= 4 and h >= 0.65 * base_height:
        return {"type": "stem", "text": "ा", "conf": 0.99, "strategy": "stem"}

    pred_full = predict_glyph(full_model, full_classes, full_map, crop, base_height)
    if pred_full["conf"] >= threshold:
        pred_full.update({"type": "full", "strategy": "cascade_pass"})
        return pred_full

    pred_half = predict_glyph(half_model, half_classes, half_map, crop, base_height)
    pred_comp = predict_glyph(comp_model, comp_classes, comp_map, crop, base_height)

    if comp["aspect_ratio"] < 0.70 and pred_half["conf"] >= 0.35:
        pred_half.update({"type": "half", "strategy": "fallback_half"})
        return pred_half

    if pred_comp["conf"] > pred_half["conf"]:
        pred_comp.update({"type": "compound", "strategy": "fallback_compound"})
        return pred_comp
    else:
        pred_half.update({"type": "half", "strategy": "fallback_half"})
        return pred_half

def reconstruct_word_text(mid_preds, up_preds, low_preds):
    if not mid_preds:
        return ""

    syllables = [{"mid": m, "upper": [], "lower": [], "cx": m["comp"]["cx"]} for m in mid_preds]

    for u in up_preds:
        u_cx = u["comp"]["cx"]
        if "i_matra" in u["class"]:
            idx = 0
            best_dist = 9999
            for i, syl in enumerate(syllables):
                d = syl["cx"] - u_cx
                if -5 <= d < best_dist:
                    best_dist = d
                    idx = i
            syllables[idx]["upper"].append(u)
        else:
            idx = min(range(len(syllables)), key=lambda i: abs(syllables[i]["cx"] - u_cx))
            syllables[idx]["upper"].append(u)

    for l in low_preds:
        idx = min(range(len(syllables)), key=lambda i: abs(syllables[i]["cx"] - l["comp"]["cx"]))
        syllables[idx]["lower"].append(l)

    word = ""
    for syl in syllables:
        m = syl["mid"]
        if m["type"] == "stem":
            if word and not word.endswith("ा"):
                word += "ा"
            else:
                word += m["text"]
            continue

        chunk = ""
        reph = any("reph" in u["class"] for u in syl["upper"])
        if reph: chunk += "र्"

        chunk += m["text"]

        for l in syl["lower"]:
            chunk += "़" if "nukta" in l["class"] else ("्र" if "rakar" in l["class"] else l["text"])

        non_reph = [u for u in syl["upper"] if "reph" not in u["class"]]
        for u in non_reph:
            if "anusvara" not in u["class"] and "chandrabindu" not in u["class"]:
                chunk += u["text"]
        for u in non_reph:
            if "anusvara" in u["class"] or "chandrabindu" in u["class"]:
                chunk += u["text"]

        word += chunk

    return unicodedata.normalize("NFC", word)

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

df_test = pd.read_parquet("dataset/data/test.parquet")
print("=== RUNNING SAMPLE EVALUATION ===")
for i in range(10):
    row = df_test.iloc[i]
    raw = row.get("image.bytes", row.get("image_bytes"))
    nparr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    seg = segment_word_image(img)
    base_h = seg["base_height"]

    up_preds = []
    for c in seg["upper"]:
        p = predict_glyph(upper_model, upper_classes, upper_map, c["crop"], base_h)
        p["comp"] = c
        up_preds.append(p)

    low_preds = []
    for c in seg["lower"]:
        p = predict_glyph(lower_model, lower_classes, lower_map, c["crop"], base_h)
        p["comp"] = c
        low_preds.append(p)

    mid_preds = []
    for c in seg["middle"]:
        p = route_middle_component(c, base_h)
        p["comp"] = c
        mid_preds.append(p)

    pred = reconstruct_word_text(mid_preds, up_preds, low_preds)
    gt = row["label"]
    cer = compute_cer(gt, pred)
    acc = max(0.0, 1.0 - cer) * 100.0

    print(f"\n--- Sample {i} ({row['filename']}) ---")
    print(f"GT:   {gt}")
    print(f"PRED: {pred} (Acc: {acc:.1f}%, CER: {cer:.2f})")
    print("Middle components:", [(m['text'], m['class'] if 'class' in m else m['type'], round(m['conf'], 2), m['strategy']) for m in mid_preds])
    print("Upper components:", [(u['text'], u['class'], round(u['conf'], 2)) for u in up_preds])
    print("Lower components:", [(l['text'], l['class'], round(l['conf'], 2)) for l in low_preds])
