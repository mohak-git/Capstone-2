# Lower Modifier Recognition Model

3-stage CNN (64x64 single-channel) for Devanagari vowel descenders and diacritics below the baseline.

## Classes (7 Total)
- `01_u_matra` (ु), `02_uu_matra` (ू), `03_vocalic_r` (ृ), `04_vocalic_rr` (ॄ), `05_halant` (्), `06_nukta` (़), `07_rakar` (्र)

## Files
- `01_generate_dataset.py` - Synthetic generator with consonant anchoring
- `02_train_model.py` - CNN training script (Adam, CrossEntropy)
- `classes.json` - Class codepoints & labels
- `lower_modifier.pth` - Saved weights

## Commands
```bash
python model/lower_modifier/01_generate_dataset.py
python model/lower_modifier/02_train_model.py --epochs 20 --batch-size 64
```
