import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt
import gzip
import time


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
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


def create_histogram(data, bins=10, output_path="histogram.png", title="Histogram",
                     xlabel="Value", percentage=True):
    plt.figure(figsize=(8, 6))

    if len(data) > 0:
        if percentage:
            weights = np.ones(len(data), dtype=np.float64) * (100.0 / len(data))
            plt.hist(data, bins=bins, range=(0, 100),
                     weights=weights, color='blue', alpha=0.7, edgecolor='black')
            plt.ylabel("Samples (%)")
            plt.ylim(0, 100)
        else:
            plt.hist(data, bins=bins, color='green', alpha=0.7, edgecolor='black')
            plt.ylabel("Count")
    else:
        plt.text(0.5, 0.5, "אין נתונים", ha='center', va='center',
                 fontsize=14, color='red')

    plt.title(f"{title} (n={len(data)})")
    plt.xlabel(xlabel)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def run_inference(session, input_data):
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data})
    return outputs[0]


def run_all_imgs(model_path, images, labels):
    session = ort.InferenceSession(model_path)
    # session = ort.InferenceSession(
    #     model_path,
    #     providers=["OpenVINOExecutionProvider", "CPUExecutionProvider"]
    # )

    # session = ort.InferenceSession(
    #     model_path,
    #     providers=["OpenVINOExecutionProvider", "CPUExecutionProvider"]
    # )
    # print("Using provider:", session.get_providers())

    # session = ort.InferenceSession(
    #     model_path,
    #     providers=[("OpenVINOExecutionProvider", {"device_type": "GPU"})]
    # )

    print("Providers in use:", session.get_providers())

    correct_probs = []
    incorrect_probs = []
    times = []

    total = len(images)

    for i in range(total):
        img = images[i].astype(np.float32) / 255.0
        img = img.reshape(1, 1, 28, 28)

        start_time = time.time()
        output = run_inference(session, img)
        end_time = time.time()

        times.append(end_time - start_time)

        logits = output.squeeze()
        probs = softmax_np(logits)
        predicted_label = int(np.argmax(logits))
        actual_label = int(labels[i])

        if predicted_label == actual_label:
            correct_probs.append(float(probs[predicted_label] * 100.0))
        else:
            incorrect_probs.append(float(probs[predicted_label] * 100.0))

    avg_time = np.mean(times)
    return correct_probs, incorrect_probs, times, avg_time


if __name__ == "__main__":
    model_path = "mnist_emnist_blank_cnn_v1.onnx"

    images_path = "t10k-images-idx3-ubyte-with-empty.gz"
    labels_path = "t10k-labels-idx1-ubyte-with-empty.gz"

    images, labels = load_mnist_dataset(images_path, labels_path)
    correct, incorrect, times, avg_time = run_all_imgs(model_path, images, labels)

    # המרה למיקרו-שניות
    times_us = [t * 1e6 for t in times]
    avg_time_us = avg_time * 1e6

    create_histogram(correct, bins=10, output_path="hist_success_new_model.png",
                     title="Histogram of Successes", xlabel="Probability (%)", percentage=True)
    create_histogram(incorrect, bins=10, output_path="hist_failures_new_model.png",
                     title="Histogram of Failures", xlabel="Probability (%)", percentage=True)
    create_histogram(times_us, bins=30, output_path="hist_times_new_model.png",
                     title="Histogram of Inference Times", xlabel="Time (µs)", percentage=False)

    print("מספר הצלחות:", len(correct))
    print("מספר כשלונות:", len(incorrect))
    print("זמן ממוצע להרצת תמונה:", avg_time_us, "מיקרו־שניות")
