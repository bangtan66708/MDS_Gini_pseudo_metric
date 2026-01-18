import urllib.request
import gzip
import shutil
from pathlib import Path

URLS = {
    "train_images": "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
}

DATA_DIR = Path("data/mnist")
DATA_DIR.mkdir(parents=True, exist_ok=True)

for name, url in URLS.items():
    gz_path = DATA_DIR / f"{name}.gz"
    raw_path = DATA_DIR / name

    if raw_path.exists():
        continue

    print(f"Download {name}...")
    urllib.request.urlretrieve(url, gz_path)

    with gzip.open(gz_path, "rb") as f_in:
        with open(raw_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    gz_path.unlink()

print("MNIST download in ./data/mnist/")
