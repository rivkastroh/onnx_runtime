import gzip
import numpy as np
from PIL import Image, ImageFilter
import struct


def load_mnist_images(filename):
    """
    טוען קובץ תמונות MNIST בפורמט gzip
    """
    with gzip.open(filename, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)
    return images, num_images, rows, cols


def save_mnist_images(filename, images):
    """
    שומר קובץ תמונות MNIST בפורמט gzip
    """
    num_images, rows, cols = images.shape
    with gzip.open(filename, "wb") as f:
        header = struct.pack(">IIII", 2051, num_images, rows, cols)
        f.write(header)
        f.write(images.astype(np.uint8).tobytes())


def blur_images(images, blur_radius=4):
    """
    מטשטש את התמונות באמצעות Gaussian blur
    """
    blurred = []
    for img_arr in images:
        img = Image.fromarray(img_arr, mode="L")
        blurred_img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        blurred.append(np.array(blurred_img, dtype=np.uint8))
    return np.stack(blurred, axis=0)


if __name__ == "__main__":
    input_images_path = "t10k-images-idx3-ubyte.gz"   # הקובץ המקורי
    output_images_path = "t10k-images-blurred.gz"     # הקובץ החדש (מטושטש)

    images, num_images, rows, cols = load_mnist_images(input_images_path)
    blurred_images = blur_images(images, blur_radius=2)
    save_mnist_images(output_images_path, blurred_images)

    print(f"קובץ חדש עם {num_images} תמונות מטושטשות נשמר כ: {output_images_path}")
