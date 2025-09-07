import gzip
import struct
import random
import numpy as np

# File names
images_file = "train-images-idx3-ubyte.gz"
labels_file = "train-labels-idx1-ubyte.gz"
out_images_file = "train-images-idx3-ubyte-with-empty.gz"
out_labels_file = "train-labels-idx1-ubyte-with-empty.gz"

def read_images(file_path):
    with gzip.open(file_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows, cols)
    return images, rows, cols

def read_labels(file_path):
    with gzip.open(file_path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def write_images(file_path, images):
    num = images.shape[0]
    rows = images.shape[1]
    cols = images.shape[2]
    with gzip.open(file_path, 'wb') as f:
        f.write(struct.pack(">IIII", 2051, num, rows, cols))
        f.write(images.tobytes())

def write_labels(file_path, labels):
    num = labels.shape[0]
    with gzip.open(file_path, 'wb') as f:
        f.write(struct.pack(">II", 2049, num))
        f.write(labels.tobytes())

def create_noisy_empty_images(num, rows, cols, noise_level=60):
    # תמונות ריקות עם רעש חלש בגווני אפור (0-60)
    return np.random.randint(0, noise_level + 1, size=(num, rows, cols), dtype=np.uint8)

def main():
    images, rows, cols = read_images(images_file)
    labels = read_labels(labels_file)

    # Create 1000 noisy empty images
    empty_images = create_noisy_empty_images(1000, rows, cols, noise_level=60)
    empty_labels = np.full(1000, 10, dtype=np.uint8) # label 10 for 'empty'

    # Concatenate
    all_images = np.concatenate([images, empty_images], axis=0)
    all_labels = np.concatenate([labels, empty_labels], axis=0)

    # Shuffle together
    idx = np.arange(all_images.shape[0])
    random.seed(42)
    np.random.seed(42)
    np.random.shuffle(idx)
    shuffled_images = all_images[idx]
    shuffled_labels = all_labels[idx]

    write_images(out_images_file, shuffled_images)
    write_labels(out_labels_file, shuffled_labels)

if __name__ == "__main__":
    main()