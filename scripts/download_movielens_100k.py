from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile


DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "dataset"
ZIP_PATH = DATASET_DIR / "ml-100k.zip"


def main():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    if (DATASET_DIR / "ml-100k").exists():
        print("MovieLens 100K already exists at:", DATASET_DIR / "ml-100k")
        return

    print("Downloading MovieLens 100K...")
    urlretrieve(DATASET_URL, ZIP_PATH)

    print("Extracting dataset...")
    with ZipFile(ZIP_PATH, "r") as zip_file:
        zip_file.extractall(DATASET_DIR)

    ZIP_PATH.unlink()
    print("Done:", DATASET_DIR / "ml-100k")


if __name__ == "__main__":
    main()
