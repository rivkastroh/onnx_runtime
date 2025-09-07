# run model of 80 img, if output right save percent, create istogram

import onnxruntime as ort, numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import gzip

def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.max(x)          # יציבות מספרית
    e = np.exp(x)
    return e / e.sum()


def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        # 4 בייטים ראשונים = magic number, 4 הבאים = מספר פריטים
        magic, num_items = np.frombuffer(f.read(8), dtype='>i4')
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num_images, rows, cols = np.frombuffer(f.read(16), dtype='>i4')
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)
    return images


def load_mnist_dataset(images_path, labels_path):
    images = load_mnist_images(images_path)
    labels = load_mnist_labels(labels_path)
    return images, labels

def create_histogram(data_percent, bins=10, output_path="histogram.png"):
    # משקולות כך שסכום העמודות יהיה 100%
    weights = np.ones(len(data_percent), dtype=np.float64) * (100.0 / len(data_percent))

    hist, bin_edges = np.histogram(data_percent, bins=bins, range=(0, 100), weights=weights)

    plt.figure(figsize=(8, 6))
    plt.hist(data_percent, bins=bins, range=(0, 100),
             weights=weights, color='blue', alpha=0.7, edgecolor='black')
    plt.title(f"Histogram of correct-class probabilities (n={len(data_percent)})")
    plt.xlabel("Probability (%)")
    plt.ylabel("Samples (%)")   # מנורמל לאחוז דגימות
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    return hist, bin_edges


def run_inference(model_path, input_data):
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data})
    print("-------------------")
    print(outputs)
    return outputs[0]

def run_all_imgs(model_path, images, labels):
    correct = 0
    total = len(images)
    results = []

    for i in range(total):
        img = images[i].astype(np.float32) / 255.0  # Normalize to [0, 1]
        img = img.reshape(1, 1, 28, 28)  # Reshape to (1, 1, 28, 28)
        
        output = run_inference(model_path, img)
        logits = output.squeeze()
        predicted_label = int(np.argmax(logits))
        actual_label = int(labels[i])
        
        if predicted_label == actual_label:
            correct += 1
            probs = softmax_np(logits)            # המרה להסתברויות
            results.append(float(probs[predicted_label]*100.0))  # Correct prediction
        
    #accuracy = correct / total * 100
    #return accuracy, results
    return results

if __name__ == "__main__":
    #model_path = "mnist.onnx"
    model_path = "mnist_emnist_blank_cnn_v1.onnx"
    images_path = "custom-images-idx3-ubyte.gz"
    labels_path = "custom-labels-idx1-ubyte.gz"

    images, labels = load_mnist_dataset(images_path, labels_path)
    results = run_all_imgs(model_path, images[:80], labels[:80])  # Run on first 80 images

    histogram, bin_edges = create_histogram(results, bins=10, output_path="histogram_new_model.png")
    print("Histogram:", histogram)
    print("Bin edges:", bin_edges)
