"""
Unified Devanagari Font Manager.

Manages all Devanagari fonts used across models (half-character, lower-modifier, upper-modifier).
Downloads missing fonts once into 'model/fonts'. If fonts are already present, they are reused.
"""

import os
import shutil
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(BASE_DIR, "fonts")

FONTS = {
    # Core Devanagari Fonts
    "NotoSans-Regular.ttf": "https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf",
    "NotoSans-Bold.ttf": "https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Bold.ttf",
    "NotoSerif-Regular.ttf": "https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSerifDevanagari/NotoSerifDevanagari-Regular.ttf",
    "NotoSerif-Bold.ttf": "https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSerifDevanagari/NotoSerifDevanagari-Bold.ttf",
    # Handwriting & Calligraphic Fonts
    "Kalam-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/kalam/Kalam-Regular.ttf",
    "Kalam-Bold.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/kalam/Kalam-Bold.ttf",
    "Kalam-Light.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/kalam/Kalam-Light.ttf",
    "Dekko-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/dekko/Dekko-Regular.ttf",
    "Amita-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/amita/Amita-Regular.ttf",
    "Amita-Bold.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/amita/Amita-Bold.ttf",
    "Tillana-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/tillana/Tillana-Regular.ttf",
    "Tillana-Bold.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/tillana/Tillana-Bold.ttf",
    "Laila-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/laila/Laila-Regular.ttf",
    "Laila-Bold.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/laila/Laila-Bold.ttf",
    "Baloo2-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/baloo2/Baloo2%5Bwght%5D.ttf",
    "Sahitya-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/sahitya/Sahitya-Regular.ttf",
    "Sahitya-Bold.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/sahitya/Sahitya-Bold.ttf",
    "RozhaOne-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/rozhaone/RozhaOne-Regular.ttf",
    "YatraOne-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/yatraone/YatraOne-Regular.ttf",
    "Rajdhani-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/rajdhani/Rajdhani-Regular.ttf",
    "Rajdhani-Bold.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/rajdhani/Rajdhani-Bold.ttf",
    "Palanquin-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/palanquin/Palanquin-Regular.ttf",
    "Palanquin-Bold.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/palanquin/Palanquin-Bold.ttf",
    "Sarala-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/sarala/Sarala-Regular.ttf",
    "Glegoo-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/glegoo/Glegoo-Regular.ttf",
    "Glegoo-Bold.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/glegoo/Glegoo-Bold.ttf",
    "Eczar-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/eczar/Eczar%5Bwght%5D.ttf",
    "AnekDevanagari-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/anekdevanagari/AnekDevanagari%5Bwdth%2Cwght%5D.ttf",
    # Evaluation & Test Fonts
    "Mukta-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/mukta/Mukta-Regular.ttf",
    "Karma-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/karma/Karma-Regular.ttf",
    "Kurale-Regular.ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/kurale/Kurale-Regular.ttf",
}


def download_fonts():
    os.makedirs(FONT_DIR, exist_ok=True)
    downloaded = 0
    skipped = 0

    for name, url in FONTS.items():
        path = os.path.join(FONT_DIR, name)
        if os.path.exists(path) and os.path.getsize(path) > 5000:
            skipped += 1
            continue

        try:
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(url, path)
            print(f"[OK] {name}")
            downloaded += 1
        except Exception as e:
            print(f"[WARN] Failed to download {name}: {e}")
            if os.path.exists(path):
                os.remove(path)

    # Copy system font if available on Windows
    win_font = r"C:\Windows\Fonts\Nirmala.ttc"
    if os.path.exists(win_font) and not os.path.exists(os.path.join(FONT_DIR, "Nirmala.ttc")):
        try:
            shutil.copy2(win_font, FONT_DIR)
        except Exception:
            pass

    fonts = [f for f in os.listdir(FONT_DIR) if f.lower().endswith((".ttf", ".otf", ".ttc"))]
    print(f"\nFonts ready in '{FONT_DIR}': {len(fonts)} (New: {downloaded}, Existing: {skipped})")
    return fonts


if __name__ == "__main__":
    download_fonts()

