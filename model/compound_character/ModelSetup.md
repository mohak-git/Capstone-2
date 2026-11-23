# Compound Character (Conjunct) Recognition Model

Lightweight 3-stage CNN (64x64 single-channel) for fused and stacked Devanagari ligatures.

## Classes (21 Total)
- `01_ksha` (क्ष), `02_tra` (त्र), `03_jnya` (ज्ञ), `04_shra` (श्र), `05_kra` (क्र), `06_gra` (ग्र), `07_pra` (प्र), `08_bra` (ब्र), `09_nka` (ङ्क), `10_nga` (ङ्ग), `11_ddha` (द्ध), `12_dya` (द्य), `13_dva` (द्व), `14_shta` (ष्ट), `15_shtha` (ष्ठ), `16_hma` (ह्म), `17_hya` (ह्य), `18_kta` (क्त), `19_tta` (त्त), `20_shca` (श्च), `21_shva` (श्व)

## Files
- `01_generate_dataset.py` - Synthetic dataset generator
- `02_train_model.py` - Training script (Adam, CrossEntropy)
- `classes.json` - Class definitions & Unicode mappings
- `compound_character.pth` - Trained model weights

## Commands
```bash
python 01_generate_dataset.py
python 02_train_model.py --epochs 20 --batch-size 64
```
