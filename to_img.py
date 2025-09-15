import os
import gzip
import struct
import numpy as np
from PIL import Image

def read_idx_images(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num, rows, cols)
        return num, rows, cols, data

def read_idx_labels(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return num, labels

def save_images_to_folder(images, labels, folder):
    os.makedirs(folder, exist_ok=True)
    for i, (img, lbl) in enumerate(zip(images, labels)):
        im = Image.fromarray(img)  # לא צריך mode
        im.save(os.path.join(folder, f"{i:05d}-{lbl}.png"))

def main():
    num_img, rows, cols, images = read_idx_images("train-images-idx3-ubyte-with-empty.gz")
    num_lbl, labels = read_idx_labels("train-labels-idx1-ubyte-with-empty.gz")

    assert num_img == num_lbl, "מספר התמונות והתוויות לא תואם"

    save_images_to_folder(images, labels, "img_with_empty")

if __name__ == "__main__":
    main()
