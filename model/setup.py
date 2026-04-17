from pathlib import Path
import zipfile, shutil, subprocess, sys


def main():
    BASE_DIR = Path(__file__).parent
    CLASSES_DIR = BASE_DIR / "classes"
    ZIP_PATH = CLASSES_DIR / "archive.zip"

    # 1. Find ZIP
    if not ZIP_PATH.exists():
        print(f"Error: No ZIP in {CLASSES_DIR}. Download archive.zip from Kaggle.")
        return

    # 2. Extract & Rename
    print(f"Processing {ZIP_PATH.name}...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(CLASSES_DIR)

    for d in ["train", "test"]:
        src, dst = CLASSES_DIR / d, CLASSES_DIR / f"{d}ing"
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            src.rename(dst)

    # 3. Preprocess & Clean
    print("Preparing classes...")
    subprocess.run([sys.executable, "01-prepare-classes.py"], cwd=BASE_DIR, check=True)

    for d in ["training", "testing"]:
        shutil.rmtree(CLASSES_DIR / d, ignore_errors=True)

    print("\nSetup complete. Data ready in model/classes/train and test.")

    # 4. Ask for training
    choice = input("\nStart training now? (y/n): ").lower()
    if choice == "y":
        print("Training model...")
        subprocess.run([sys.executable, "02-train-model.py"], cwd=BASE_DIR, check=True)
    else:
        print("Setup finished. Training with: python model/02-train-model.py")


if __name__ == "__main__":
    main()
