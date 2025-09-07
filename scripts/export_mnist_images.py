import gzip
import struct
import numpy as np
from PIL import Image
import os

# ----------------------------------------------------------
# פונקציה לטעינת התמונות בפורמט MNIST
# ----------------------------------------------------------
def load_images(filename: str) -> np.ndarray:
    with gzip.open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows, cols)

# ----------------------------------------------------------
# שלב ראשי
# ----------------------------------------------------------
if __name__ == "__main__":
    images = load_images("t10k-images-idx3-ubyte.gz")
    output_dir = "/mnt/c/mnist_images"

    os.makedirs(output_dir, exist_ok=True)

    # לדוגמה – נשמור 100 תמונות ראשונות
    for i in range(100):
        img = Image.fromarray(images[i], mode="L")  # "L" = grayscale
        img.save(os.path.join(output_dir, f"digit_{i}.png"))

    print(f"נשמרו {i+1} תמונות אל {output_dir}")

