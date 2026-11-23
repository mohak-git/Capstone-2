# Full Character Recognition Model

3-stage CNN (64x64 single-channel) for Devanagari full consonants, independent vowels, digits, and punctuation.

## Classes (59 Total)
- **33 Consonants**: `01_ka` (क) to `33_ha` (ह)
- **3 Conjuncts**: `34_ksha` (क्ष), `35_tra` (त्र), `36_jnya` (ज्ञ)
- **11 Vowels**: `37_vowel_a` (अ) to `47_vowel_au` (औ)
- **10 Digits**: `48_digit_0` (०) to `57_digit_9` (९)
- **2 Symbols**: `58_visarga` (ः), `59_avagraha` (ऽ)

## Files
- `01_generate_dataset.py` - Synthetic generator with headline removal
- `02_train_model.py` - CNN training script (Adam, CrossEntropy)
- `classes.json` - Class codepoints & labels
- `full_character.pth` - Saved weights

## Commands
```bash
python model/full_character/01_generate_dataset.py
python model/full_character/02_train_model.py --epochs 20 --batch-size 64
```
