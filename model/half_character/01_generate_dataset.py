import json
import os
import shutil
import cv2
import numpy as np
from collections import defaultdict
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

TRAIN_N = 200
TEST_N = 50
IMG_SIZE = (64, 64)
RANDOM_SEED = 61

RESERVED_TEST_FONTS = {
    "NotoSerifDevanagari-Bold",
    "NotoSerif-Bold",
    "Mukta-Regular",
    "Sahitya-Regular",
    "Kurale-Regular",
    "Karma-Regular",
}

V, ZWJ = "्", "‍"


def gids(font, text):
    return [g[0] for g in font.shape(text)]


def ink_of(font, text):
    glyphs = [x for x in font.shape(text) if x[0] != 0 or text.endswith(ZWJ)]
    if not text.endswith(ZWJ) and any(x[0] == 0 for x in font.shape(text)):
        return None
    layers = font.raster([x for x in glyphs if x[0] != 0])
    return None if layers is None else layers.max(axis=0) > 0.5


def strip_headline(mask):
    profile = mask.sum(axis=1)
    if not profile.any():
        return mask
    threshold = 0.6 * profile.max()
    top_row = int(np.where(profile >= threshold)[0].min())
    bottom_row = top_row
    while bottom_row + 1 < len(profile) and profile[bottom_row + 1] >= threshold:
        bottom_row += 1

    if bottom_row - top_row > RENDER_SIZE // 8:
        return mask

    out = mask.copy()
    below = mask[min(bottom_row + 3, mask.shape[0] - 1)]
    above = mask[max(top_row - 3, 0)]
    for r in range(top_row, bottom_row + 1):
        out[r] &= below | above
    return out


def clean_mask(before, after):
    if after is None or after.sum() < 0.5 * before.sum():
        return None
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(after.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return None
    largest_area = stats[1:, cv2.CC_STAT_AREA].max()
    retained = np.zeros_like(after)
    for idx in range(1, num_labels):
        if stats[idx, cv2.CC_STAT_AREA] >= 0.04 * largest_area:
            retained |= labels == idx
    return retained


def crop_bounds(mask):
    if mask is None:
        return None
    ys, xs = np.where(mask)
    return mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].astype(np.uint8) if len(ys) else None


def extract_half_mask(font, cons):
    if not font.has(cons.replace(V, "")):
        return None, "missing"

    full_ink = ink_of(font, cons)
    if full_ink is None:
        return None, "missing"

    # Font must substitute a genuine half glyph (no visible halant)
    half_with_zwj = [g for g in gids(font, cons + V + ZWJ) if g not in gids(font, ZWJ) and g != 0]
    half_with_virama = gids(font, cons + V)
    virama_glyphs = set(half_with_virama) - set(gids(font, cons))

    if half_with_zwj == half_with_virama or virama_glyphs & set(half_with_zwj):
        return None, "no_half_form"

    half_ink = ink_of(font, cons + V + ZWJ)
    if half_ink is None:
        return None, "missing"

    hw, fw = np.where(half_ink.any(axis=0))[0], np.where(full_ink.any(axis=0))[0]
    hb, fb = np.where(half_ink.any(axis=1))[0].max(), np.where(full_ink.any(axis=1))[0].max()

    if (hw.max() - hw.min()) > 0.97 * (fw.max() - fw.min()) or hb > fb + RENDER_SIZE // 40:
        return None, "no_half_form"

    cleaned = crop_bounds(clean_mask(half_ink, strip_headline(half_ink)))
    return (cleaned, "half_glyph") if cleaned is not None else (None, "mostly_headline")


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

    print(f"Generating half character dataset for {len(classes)} classes...")
    print(f"Loaded: {len(train_fonts)} train fonts, {len(test_fonts)} test fonts")

    glyph_pool = {split: defaultdict(list) for split in font_splits}
    for split, font_list in font_splits.items():
        for font in tqdm(font_list, desc=f"Extracting {split} prototypes", leave=False):
            for cls in classes:
                name = cls["name"]
                base_char = cls.get("char", cls["sign"].replace(V, ""))
                mask, method = extract_half_mask(font, base_char)
                if mask is not None:
                    glyph_pool[split][name].append((mask, font.name, method))

    ink = Ink(seed=RANDOM_SEED)

    for cls in tqdm(classes, desc="Classes"):
        name = cls["name"]
        folder = cls.get("folder", f"{cls['id']:02d}_{name}")

        for split, count in (("train", TRAIN_N), ("test", TEST_N)):
            out_dir = os.path.join(roots[split], folder)
            os.makedirs(out_dir, exist_ok=True)

            prototypes = glyph_pool[split][name]
            if not prototypes:
                print(f"[WARN] No prototypes for half class: {name} in {split}")
                continue

            made, tries = 0, 0
            while made < count and tries < count * 25:
                tries += 1
                mask, fname, _ = ink.rng.choice(prototypes)
                sample_img, _ = ink.draw(mask, font_scales[fname])
                if sample_img is None:
                    continue

                cv2.imwrite(os.path.join(out_dir, f"{name}_{made:05d}.png"), sample_img)
                made += 1

    print(f"\nDataset ready: {DATA_DIR}")


if __name__ == "__main__":
    generate_dataset()
