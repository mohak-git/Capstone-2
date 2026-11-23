# Half Character Recognition Model

3-stage CNN (64x64 single-channel) for authentic stemless Devanagari half-consonants.

## Classes (26 Total)
- `01_ka` (क्), `02_kha` (ख्), `03_ga` (ग्), `04_gha` (घ्), `05_ca` (च्), `06_ja` (ज्), `07_jha` (झ्), `08_nya` (ञ्), `09_nna` (ण्), `10_ta` (त्), `11_tha` (थ्), `12_dha` (ध्), `13_na` (न्), `14_pa` (प्), `15_pha` (फ्), `16_ba` (ब्), `17_bha` (भ्), `18_ma` (म्), `19_ya` (य्), `20_la` (ल्), `21_va` (व्), `22_sha` (श्), `23_ssa` (ष्), `24_sa` (स्), `25_ksha` (क्ष्), `26_jnya` (ज्ञ्)

## Files
- `01_generate_dataset.py` - Synthetic generator with ZWJ half-form extraction
- `02_train_model.py` - CNN training script (Adam, CrossEntropy)
- `classes.json` - Class codepoints & labels
- `half_character.pth` - Saved weights

## Commands
```bash
python model/half_character/01_generate_dataset.py
python model/half_character/02_train_model.py --epochs 20 --batch-size 64
```
