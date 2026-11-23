import json
import os
import shutil
import cv2
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SHARED_FONT_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "fonts"))
FONT_DIR = SHARED_FONT_DIR if os.path.exists(SHARED_FONT_DIR) else os.path.join(BASE_DIR, "fonts")
CLASS_FILE = os.path.join(BASE_DIR, "classes.json")
DATA_DIR = os.path.join(BASE_DIR, "dataset")

MODEL_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, MODEL_ROOT)
from glyph_utils import CANVAS_SIZE, REF_HEIGHT, RENDER_SIZE, Font, Ink, scale_of

TRAIN_N = 1000
TEST_N = 200
IMG_SIZE = (64, 64)
RANDOM_SEED = 42

RESERVED_TEST_FONTS = {
    "NotoSerifDevanagari-Bold",
    "NotoSerif-Bold",
    "Mukta-Regular",
    "Sahitya-Regular",
    "Kurale-Regular",
    "Karma-Regular",
}

# Diverse structural consonant anchors (central-stem, right-stem, round, loop, and stemless)
REPRESENTATIVE_CONSONANTS = list("कगचटतपदमह")
MINIMAL_CONSONANTS = list("कपद")


def extract_upper_mask(font, base_char, sign, preserve_stem):
    full_text = ("र्" + base_char) if sign == "REPH" else (base_char + sign)
    clean_check = full_text.replace("्", "")
    if not font.has(clean_check):
        return None

    full_glyphs = font.shape(full_text)
    plain_glyphs = font.shape(base_char)
    if any(g[0] == 0 for g in full_glyphs):
        return None

    need = Counter(g[0] for g in plain_glyphs)
    have = Counter(g[0] for g in full_glyphs)
    if any(have[k] < v for k, v in need.items()):
        return None

    extra = have - need
    is_sign = []
    for g in full_glyphs:
        is_sign.append(extra[g[0]] > 0)
        if is_sign[-1]:
            extra[g[0]] -= 1
    is_sign = np.array(is_sign)
    if not is_sign.any() or is_sign.all():
        return None

    layers = font.raster(full_glyphs)
    if layers is None:
        return None

    sign_ink = layers[is_sign].max(axis=0) > 0.5
    base_ink = layers[~is_sign].max(axis=0) > 0.5

    row_profile = base_ink.sum(axis=1)
    if not row_profile.any():
        return None
    threshold = 0.6 * row_profile.max()
    top_row = int(np.where(row_profile >= threshold)[0].min())
    bottom_row = top_row
    while bottom_row + 1 < len(row_profile) and row_profile[bottom_row + 1] >= threshold:
        bottom_row += 1

    mask = sign_ink.copy()
    if preserve_stem:
        below_edge = mask[min(bottom_row + 3, mask.shape[0] - 1)]
        above_edge = mask[max(top_row - 3, 0)]
        for r in range(top_row, bottom_row + 1):
            mask[r] &= below_edge | above_edge
    else:
        cutoff_row = max(0, top_row - RENDER_SIZE // 60)
        mask[cutoff_row:, :] = False
        if mask.sum() < 0.55 * sign_ink.sum():
            return None

    if mask.sum() == 0:
        return None

    ys, xs = np.where(mask)
    return mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].astype(np.uint8)


def load_fonts(font_dir):
    fonts = []
    for filename in sorted(os.listdir(font_dir)):
        if filename.lower().endswith((".ttf", ".otf")):
            try:
                fonts.append(Font(os.path.join(font_dir, filename)))
            except Exception as err:
                print(f"Skip font {filename}: {err}")
    return fonts


def generate_dataset():
    if not os.path.exists(FONT_DIR) or not os.listdir(FONT_DIR):
        print(f"Error: No fonts found in '{FONT_DIR}'. Run 'python model/download_fonts.py' first.")
        return

    with open(CLASS_FILE, encoding="utf-8") as f:
        classes = json.load(f)

    fonts = load_fonts(FONT_DIR)
    train_fonts = [f for f in fonts if f.name not in RESERVED_TEST_FONTS]
    test_fonts = [f for f in fonts if f.name in RESERVED_TEST_FONTS]
    if not test_fonts:
        test_fonts, train_fonts = fonts[-3:], fonts[:-3]

    font_scales = {f.name: scale_of(f) for f in fonts}
    font_splits = {"train": train_fonts, "test": test_fonts}

    roots = {
        "train": os.path.join(DATA_DIR, "train"),
        "test": os.path.join(DATA_DIR, "test"),
    }
    for r in roots.values():
        os.makedirs(r, exist_ok=True)

    print(f"Generating upper modifier dataset for {len(classes)} classes...")
    print(f"Loaded: {len(train_fonts)} train fonts, {len(test_fonts)} test fonts")

    glyph_pool = {split: defaultdict(list) for split in font_splits}
    for split, font_list in font_splits.items():
        for font in tqdm(font_list, desc=f"Extracting {split} prototypes", leave=False):
            for cls in classes:
                name = cls["name"]
                sign = cls["sign"]
                preserve_stem = cls.get("with_stem", False)
                base_candidates = REPRESENTATIVE_CONSONANTS if preserve_stem or sign == "REPH" else MINIMAL_CONSONANTS
                seen = set()
                for base in base_candidates:
                    mask = extract_upper_mask(font, base, sign, preserve_stem)
                    if mask is None:
                        continue
                    sig = (mask.shape, int(mask.sum()))
                    if sig in seen:
                        continue
                    seen.add(sig)
                    glyph_pool[split][name].append((mask, font.name, base))

    ink = Ink(seed=RANDOM_SEED)

    for cls in tqdm(classes, desc="Classes"):
        name = cls["name"]
        folder = cls.get("folder", f"{cls['id']:02d}_{name}")

        for split, count in (("train", TRAIN_N), ("test", TEST_N)):
            out_dir = os.path.join(roots[split], folder)
            os.makedirs(out_dir, exist_ok=True)

            font_groups = defaultdict(list)
            for item in glyph_pool[split][name]:
                font_groups[item[1]].append(item)

            font_names = sorted(font_groups.keys())
            if not font_names:
                print(f"[WARN] No prototypes for class: {name} in {split}")
                continue

            made, tries = 0, 0
            while made < count and tries < count * 25:
                tries += 1
                picked_font = ink.rng.choice(font_names)
                mask, fname, base_char = ink.rng.choice(font_groups[picked_font])

                sample_img, _ = ink.draw(mask, font_scales[fname])
                if sample_img is None:
                    continue

                cv2.imwrite(os.path.join(out_dir, f"{name}_{made:05d}.png"), sample_img)
                made += 1

    print(f"\nDataset ready: {DATA_DIR}")


if __name__ == "__main__":
    generate_dataset()
