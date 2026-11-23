# Upper Modifier Recognition Model

3-stage CNN (64x64 single-channel) for Devanagari vowel ascenders, bindu, and reph above the headline.

## Classes (11 Total)
- `01_i_matra` (ि), `02_ii_matra` (ी), `03_e_matra` (े), `04_ai_matra` (ै), `05_o_matra` (ो), `06_au_matra` (ौ), `07_candra_e` (ॅ), `08_candra_o` (ॉ), `09_anusvara` (ं), `10_chandrabindu` (ँ), `11_reph` (र्)

## Files
- `01_generate_dataset.py` - Synthetic generator with headline separation
- `02_train_model.py` - CNN training script (Adam, CrossEntropy)
- `classes.json` - Class codepoints & labels
- `upper_modifier.pth` - Saved weights

## Commands
```bash
python model/upper_modifier/01_generate_dataset.py
python model/upper_modifier/02_train_model.py --epochs 20 --batch-size 64
```
