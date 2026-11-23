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
BASE_MAP = {
    "nukta": list("कखगजडढफय"),
    "rakar": list("टठडढदछ"),
}


def render_placed_glyphs(font, glyphs, canvas_box):
    min_x, min_y, width, height = canvas_box
    canvas = np.zeros((height, width), np.float32)

    for gid, x_pos, y_pos in glyphs:
        bitmap, left, top = font._render_glyph(gid)
        if not bitmap.size:
            continue

        px = round(x_pos / 64) + left - min_x
        py = -round(y_pos / 64) - top - min_y
        row_slice = slice(max(0, py), min(height, py + bitmap.shape[0]))
        col_slice = slice(max(0, px), min(width, px + bitmap.shape[1]))

        sub_bmp = bitmap[row_slice.start - py : row_slice.stop - py, col_slice.start - px : col_slice.stop - px] / 255.0
        canvas[row_slice, col_slice] = np.maximum(canvas[row_slice, col_slice], sub_bmp)
    return canvas


def extract_lower_ink(sign_ink, base_ink):
    active_rows = np.where(base_ink.any(axis=1))[0]
    if not len(active_rows):
        return None

    lower_ink = sign_ink.copy()
    cutoff_row = active_rows.max() - RENDER_SIZE // 80
    lower_ink[:cutoff_row, :] = False

    if lower_ink.sum() == 0 or lower_ink.sum() < 0.55 * sign_ink.sum():
        return None

    ys, xs = np.where(lower_ink)
    return lower_ink[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].astype(np.uint8)


def extract_nukta_mask(font, base_char):
    full_glyphs = font.shape(base_char + "़")
    plain_glyphs = font.shape(base_char)
    need = Counter(g[0] for g in plain_glyphs)
    have = Counter(g[0] for g in full_glyphs)
    render_box = (-RENDER_SIZE, -2 * RENDER_SIZE, 4 * RENDER_SIZE, 4 * RENDER_SIZE)

    if all(have[k] >= v for k, v in need.items()) and (have - need):
        extra = have - need
        selected_glyphs = []
        for g in full_glyphs:
            if extra[g[0]] > 0:
                selected_glyphs.append(g)
                extra[g[0]] -= 1
        ink_mask = render_placed_glyphs(font, selected_glyphs, render_box) > 0.5
    else:
        full_canvas = render_placed_glyphs(font, full_glyphs, render_box) > 0.5
        plain_canvas = render_placed_glyphs(font, plain_glyphs, render_box) > 0.5
        if plain_canvas.sum() == 0 or (full_canvas & plain_canvas).sum() / plain_canvas.sum() < 0.93:
            return None

        dilated_base = cv2.dilate(plain_canvas.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        diff_mask = full_canvas & ~dilated_base
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(diff_mask.astype(np.uint8), connectivity=8)
        if num_labels <= 1:
            return None
        largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        ink_mask = labels == largest_idx

    ys, xs = np.where(ink_mask)
    if not len(ys):
        return None
    return ink_mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def extract_lower_mask(font, base_char, sign):
    if sign == "़":
        nukta_ink = extract_nukta_mask(font, base_char)
        if nukta_ink is None:
            return None, "missing"
        diameter = int(max(nukta_ink.shape))
        disc = np.zeros((diameter + 2, diameter + 2), np.uint8)
        cv2.circle(disc, (diameter // 2 + 1, diameter // 2 + 1), diameter // 2, 1, -1)
        return disc, "dot"

    text = base_char + sign
    if not font.has(text):
        return None, "missing"

    full_glyphs = font.shape(text)
    plain_glyphs = font.shape(base_char)
    if any(g[0] == 0 for g in full_glyphs):
        return None, "missing"

    need = Counter(g[0] for g in plain_glyphs)
    have = Counter(g[0] for g in full_glyphs)

    if all(have[k] >= v for k, v in need.items()):
        extra = have - need
        is_sign = []
        for g in full_glyphs:
            is_sign.append(extra[g[0]] > 0)
            if is_sign[-1]:
                extra[g[0]] -= 1
        is_sign = np.array(is_sign)

        if is_sign.any() and not is_sign.all():
            layers = font.raster(full_glyphs)
            mask = extract_lower_ink(layers[is_sign].max(0) > 0.5, layers[~is_sign].max(0) > 0.5)
            return (mask, "glyph") if mask is not None else (None, "inside")

    render_box = (-RENDER_SIZE, -2 * RENDER_SIZE, 4 * RENDER_SIZE, 4 * RENDER_SIZE)
    full_canvas = render_placed_glyphs(font, full_glyphs, render_box) > 0.5
    plain_canvas = render_placed_glyphs(font, plain_glyphs, render_box) > 0.5
    if plain_canvas.sum() == 0:
        return None, "missing"

    surviving_ratio = (full_canvas & plain_canvas).sum() / plain_canvas.sum()
    if surviving_ratio < 0.93:
        return None, "letter_changed"

    dilated_base = cv2.dilate(plain_canvas.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    diff_mask = full_canvas & ~dilated_base
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(diff_mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return None, "no_diff"

    retained_diff = np.zeros_like(diff_mask)
    max_area = stats[1:, cv2.CC_STAT_AREA].max()
    for idx in range(1, num_labels):
        if stats[idx, cv2.CC_STAT_AREA] >= 0.15 * max_area:
            retained_diff |= labels == idx

    mask = extract_lower_ink(retained_diff, plain_canvas)
    return (mask, "difference") if mask is not None else (None, "inside")


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

    print(f"Generating lower modifier dataset for {len(classes)} classes...")
    print(f"Loaded: {len(train_fonts)} train fonts, {len(test_fonts)} test fonts")

    glyph_pool = {split: defaultdict(list) for split in font_splits}
    for split, font_list in font_splits.items():
        for font in tqdm(font_list, desc=f"Extracting {split} prototypes", leave=False):
            for cls in classes:
                name = cls["name"]
                sign = cls["sign"]
                base_chars = BASE_MAP.get(name, REPRESENTATIVE_CONSONANTS)
                seen = set()
                for base in base_chars:
                    mask, method = extract_lower_mask(font, base, sign)
                    if mask is None:
                        continue
                    sig = (mask.shape, int(mask.sum()))
                    if sig in seen:
                        continue
                    seen.add(sig)
                    glyph_pool[split][name].append((mask, font.name, base, method))

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
                mask, fname, base_char, method = ink.rng.choice(font_groups[picked_font])

                sample_img, _ = ink.draw(mask, font_scales[fname])
                if sample_img is None:
                    continue

                cv2.imwrite(os.path.join(out_dir, f"{name}_{made:05d}.png"), sample_img)
                made += 1

    print(f"\nDataset ready: {DATA_DIR}")


if __name__ == "__main__":
    generate_dataset()
