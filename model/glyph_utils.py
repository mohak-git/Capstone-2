"""
Devanagari Modifier Glyph Utilities.

Shared across model generators and inference pipelines:
- Font shaping (HarfBuzz) and glyph rasterisation (FreeType)
- Handwriting and scanned page simulation (Ink)
- Inference crop normalization (normalize_crop)
"""

import os
import random
import cv2
import numpy as np
import freetype
import uharfbuzz as hb

CANVAS_SIZE = 64  # Output square dimensions (64x64)
REF_HEIGHT = 34   # Reference height: plain consonant 'न' is scaled to 34 px tall
RENDER_SIZE = 400 # Base rasterisation font size


class Font:
    """OpenType text shaper (HarfBuzz) and glyph rasteriser (FreeType)."""

    def __init__(self, font_path):
        self.path = font_path
        self.name = os.path.splitext(os.path.basename(font_path))[0]

        with open(font_path, "rb") as font_file:
            self.hb_font = hb.Font(hb.Face(font_file.read()))
        self.hb_font.scale = (RENDER_SIZE * 64, RENDER_SIZE * 64)

        self.ft_face = freetype.Face(font_path)
        self.ft_face.set_char_size(RENDER_SIZE * 64)
        self.glyph_cache = {}

    def has(self, text):
        """Check if font contains glyphs for all characters in text."""
        return all(self.ft_face.get_char_index(ord(char)) for char in text)

    def shape(self, text):
        """Shape text using HarfBuzz. Returns [(codepoint, x_pos, y_pos), ...]."""
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(self.hb_font, buf, {})

        glyphs, x, y = [], 0, 0
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
            glyphs.append((info.codepoint, x + pos.x_offset, y + pos.y_offset))
            x += pos.x_advance
            y += pos.y_advance
        return glyphs

    def _render_glyph(self, glyph_id):
        """Render individual glyph to bitmap array with cache."""
        if glyph_id not in self.glyph_cache:
            self.ft_face.load_glyph(glyph_id, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_NO_HINTING)
            glyph = self.ft_face.glyph
            bitmap = glyph.bitmap
            if bitmap.width and bitmap.rows:
                bitmap_arr = np.array(bitmap.buffer, np.uint8).reshape(bitmap.rows, bitmap.width)
            else:
                bitmap_arr = np.zeros((0, 0), np.uint8)
            self.glyph_cache[glyph_id] = (bitmap_arr, glyph.bitmap_left, glyph.bitmap_top)
        return self.glyph_cache[glyph_id]

    def raster(self, glyphs):
        """Composite placed glyphs onto a common canvas. Returns coverage array (layers, H, W)."""
        placed = []
        for glyph_id, x, y in glyphs:
            bitmap, left, top = self._render_glyph(glyph_id)
            placed.append((bitmap, round(x / 64) + left, -round(y / 64) - top))

        boxes = [(x, y, x + bmp.shape[1], y + bmp.shape[0]) for bmp, x, y in placed if bmp.size]
        if not boxes:
            return None

        pad = 40
        min_x = min(box[0] for box in boxes) - pad
        min_y = min(box[1] for box in boxes) - pad
        canvas_w = max(box[2] for box in boxes) + pad - min_x
        canvas_h = max(box[3] for box in boxes) + pad - min_y

        layers = np.zeros((len(placed), canvas_h, canvas_w), np.float32)
        for idx, (bmp, x, y) in enumerate(placed):
            if bmp.size:
                y_offset, x_offset = y - min_y, x - min_x
                layers[idx, y_offset : y_offset + bmp.shape[0], x_offset : x_offset + bmp.shape[1]] = bmp / 255.0
        return layers


def scale_of(font):
    """Compute scaling ratio so reference letter 'न' is exactly REF_HEIGHT pixels tall."""
    layers = font.raster(font.shape("न"))
    if layers is None:
        return 1.0
    active_rows = np.where(layers.max(0).max(1) > 0.5)[0]
    if len(active_rows) == 0:
        return 1.0
    letter_height = active_rows.max() - active_rows.min() + 1
    return REF_HEIGHT / float(letter_height)


class Ink:
    """Simulates handwritten strokes and paper scan noise on binary masks."""

    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.npr = np.random.default_rng(seed)

    def _gaussian_noise(self, shape, sigma, amplitude):
        noise = cv2.GaussianBlur(self.npr.random(shape).astype(np.float32) - 0.5, (0, 0), sigma)
        peak = np.abs(noise).max() + 1e-6
        return (noise * float(amplitude) / peak).astype(np.float32)

    def draw(self, mask, scale):
        """Synthesize handwriting on mask. Returns (64x64 uint8, param_dict) or (None, None)."""
        rng = self.rng
        pad_mask = cv2.copyMakeBorder(mask, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=0)
        mask_h, mask_w = pad_mask.shape

        # 1. Apply handwriting slant and rotation
        skew = rng.uniform(-15, 15)
        rot_deg = rng.uniform(-6, 6)
        shear = np.tan(np.radians(skew))
        angle_rad = np.radians(rot_deg)
        rot_mat = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])
        affine_2x2 = rot_mat @ np.array([[1, shear], [0, 1]])
        center = np.array([mask_w / 2, mask_h / 2])
        affine_2x3 = np.hstack([affine_2x2, (center - affine_2x2 @ center)[:, None]])
        warped_mask = cv2.warpAffine(pad_mask * 255, affine_2x3, (mask_w, mask_h)) > 127

        # 2. Pen stroke simulation with variable pressure
        pen_px = rng.uniform(2.2, 4.0)
        base_width = pen_px / scale
        pressure_map = base_width * (1 + self._gaussian_noise((mask_h, mask_w), 25, 0.35))
        canvas = np.zeros((mask_h, mask_w), np.float32)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            warped_mask.astype(np.uint8), connectivity=8
        )

        for comp_idx in range(1, num_labels):
            comp_w = stats[comp_idx, cv2.CC_STAT_WIDTH]
            comp_h = stats[comp_idx, cv2.CC_STAT_HEIGHT]

            # Render isolated dots
            if max(comp_w, comp_h) < 2.2 * base_width:
                dot_radius = max(base_width * rng.uniform(0.75, 1.15), 1.6 / scale)
                dot_center = tuple(int(val) for val in centroids[comp_idx])
                cv2.circle(canvas, dot_center, int(round(dot_radius)), 1.0, -1, cv2.LINE_AA)
                continue

            # Thin stroke to centerline and redraw with variable circular pen nib
            comp_mask = (labels == comp_idx).astype(np.uint8) * 255
            if hasattr(cv2, "ximgproc"):
                thinned = cv2.ximgproc.thinning(comp_mask)
            else:
                thinned = cv2.erode(comp_mask, np.ones((3, 3), np.uint8), iterations=1)

            ys, xs = np.where(thinned > 0)
            for y_pt, x_pt in zip(ys, xs):
                nib_radius = max(1, int(pressure_map[y_pt, x_pt] / 2))
                cv2.circle(canvas, (int(x_pt), int(y_pt)), nib_radius, 1.0, -1, cv2.LINE_AA)

        # 3. Hand tremor / wobble displacement
        wobble_amp = rng.uniform(0.6, 1.3) / scale
        wobble_sigma = 7.5 / scale
        grid_x, grid_y = np.meshgrid(np.arange(mask_w, dtype=np.float32), np.arange(mask_h, dtype=np.float32))
        disp_x = grid_x + self._gaussian_noise((mask_h, mask_w), wobble_sigma, wobble_amp)
        disp_y = grid_y + self._gaussian_noise((mask_h, mask_w), wobble_sigma, wobble_amp)
        canvas = cv2.remap(canvas, disp_x, disp_y, cv2.INTER_LINEAR)

        # 4. Crop active stroke and place into fixed canvas
        active_y, active_x = np.where(canvas > 0.3)
        if not len(active_y):
            return None, None

        crop = canvas[active_y.min() : active_y.max() + 1, active_x.min() : active_x.max() + 1]
        scaled_h = max(1, round(crop.shape[0] * scale))
        scaled_w = max(1, round(crop.shape[1] * scale))
        if max(scaled_h, scaled_w) > CANVAS_SIZE - 4:
            return None, None

        ink_layer = np.zeros((CANVAS_SIZE, CANVAS_SIZE), np.float32)
        top_offset = (CANVAS_SIZE - scaled_h) // 2
        left_offset = (CANVAS_SIZE - scaled_w) // 2
        ink_layer[top_offset : top_offset + scaled_h, left_offset : left_offset + scaled_w] = cv2.resize(
            crop, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA
        )

        # 5. Paper texture, ink absorption, sensor noise, and JPEG artifacts
        density_mod = np.clip(rng.uniform(0.75, 0.95) + self._gaussian_noise((CANVAS_SIZE, CANVAS_SIZE), 3, 0.2), 0.45, 1)
        ink_layer *= density_mod

        paper = rng.uniform(215, 242) + self._gaussian_noise((CANVAS_SIZE, CANVAS_SIZE), 12, 6)
        paper += self.npr.normal(0, 2, (CANVAS_SIZE, CANVAS_SIZE))
        ink_color = rng.uniform(25, 70)

        composite = paper * (1 - ink_layer) + ink_color * ink_layer
        composite = cv2.GaussianBlur(composite, (3, 3), rng.uniform(0.3, 0.7))
        composite += self.npr.normal(0, rng.uniform(1.5, 4), (CANVAS_SIZE, CANVAS_SIZE))
        composite = np.clip(composite, 0, 255).astype(np.uint8)

        jpeg_quality = rng.randint(60, 88)
        encoded = cv2.imencode(".jpg", composite, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])[1]
        final_img = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)

        params = dict(
            pen_px=round(pen_px, 2),
            skew=round(skew, 1),
            rot=round(rot_deg, 1),
            w=scaled_w,
            h=scaled_h,
            jpeg=jpeg_quality,
        )
        return final_img, params


def normalize_crop(gray_img, base_char_height):
    """
    Standardize segmented modifier crop to 64x64 matching dataset scale.

    Parameters:
        gray_img: Grayscale modifier image crop.
        base_char_height: Height (in px) of reference base consonant in the same line/word.

    Returns:
        64x64 uint8 numpy array with centered modifier.
    """
    scale_factor = REF_HEIGHT / float(max(1, base_char_height))
    img_h, img_w = gray_img.shape
    new_h = max(1, round(img_h * scale_factor))
    new_w = max(1, round(img_w * scale_factor))

    interpolation = cv2.INTER_AREA if scale_factor < 1 else cv2.INTER_CUBIC
    scaled_crop = cv2.resize(gray_img, (new_w, new_h), interpolation=interpolation)

    # Constrain to canvas bounds if exceeding limits
    if max(new_h, new_w) > CANVAS_SIZE:
        shrink_ratio = CANVAS_SIZE / max(new_h, new_w)
        fit_w = max(1, round(new_w * shrink_ratio))
        fit_h = max(1, round(new_h * shrink_ratio))
        scaled_crop = cv2.resize(scaled_crop, (fit_w, fit_h), interpolation=cv2.INTER_AREA)
        new_h, new_w = scaled_crop.shape

    canvas = np.full((CANVAS_SIZE, CANVAS_SIZE), 255, np.uint8)
    y_start = (CANVAS_SIZE - new_h) // 2
    x_start = (CANVAS_SIZE - new_w) // 2
    canvas[y_start : y_start + new_h, x_start : x_start + new_w] = scaled_crop
    return canvas

