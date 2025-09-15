import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt
import gzip
import time
import os
os.environ["OV_LOG_LEVEL"] = "DEBUG"



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


def run_all_imgs(model_path, images, labels, batch_size=32):   # ⚠️ הוספתי batch_size
    #session = ort.InferenceSession(model_path)
    #session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    session = ort.InferenceSession(
        model_path,
        providers=["OpenVINOExecutionProvider"],
        provider_options=[{"device_type": "GPU_FP32"}]
    )


    print("Providers in use:", session.get_providers())
    print(session.get_provider_options())

    correct_probs = []
    incorrect_probs = []
    times = []

    total = len(images)

    # ⚠️ לולאה לפי batch ולא לפי תמונה בודדת
    for i in range(0, total, batch_size):
        batch_imgs = images[i:i+batch_size].astype(np.float32) / 255.0
        batch_imgs = batch_imgs.reshape(-1, 1, 28, 28)
        batch_labels = labels[i:i+batch_size]

        start_time = time.time()
        outputs = run_inference(session, batch_imgs)  # ⚠️ מפעילים על כל ה־batch
        end_time = time.time()

        times.append((end_time - start_time) / len(batch_imgs))  # ⚠️ זמן ממוצע פר־דגימה

        # ⚠️ outputs הוא בגודל [batch, num_classes]
        for j, logits in enumerate(outputs):
            probs = softmax_np(logits)
            predicted_label = int(np.argmax(logits))
            actual_label = int(batch_labels[j])

            if predicted_label == actual_label:
                correct_probs.append(float(probs[predicted_label] * 100.0))
            else:
                incorrect_probs.append(float(probs[predicted_label] * 100.0))

    avg_time = np.mean(times)
    return correct_probs, incorrect_probs, times, avg_time

def main(batch = 1):
    #model_path = "mnist_emnist_blank_cnn_v1_quant_batch.onnx"
    #model_path = "over_models/pruned_conv3_10_fc1_30.onnx"
    model_path = "mnist_emnist_blank_cnn_v1.onnx"

    images_path =    "t10k-images-blurred.gz"   #"t10k-images-idx3-ubyte-with-empty.gz"
    labels_path =    "t10k-labels-idx1-ubyte.gz" #"t10k-labels-idx1-ubyte-with-empty.gz"

    images, labels = load_mnist_dataset(images_path, labels_path)

    # ⚠️ התחלת טיימר כולל
    total_start = time.time()

    correct, incorrect, times, avg_time = run_all_imgs(model_path, images, labels, batch_size=batch)

    # ⚠️ סיום טיימר כולל
    total_end = time.time()
    total_time = total_end - total_start
    total_time_sec = total_time
    total_time_min = total_time / 60.0

    # המרה למיקרו-שניות
    times_us = [t * 1e6 for t in times]
    avg_time_us = avg_time * 1e6

    create_histogram(correct, bins=10, output_path= f"hist_success_new_model_batch{batch}.png",
                     title="Histogram of Successes", xlabel="Probability (%)", percentage=True)
    create_histogram(incorrect, bins=10, output_path= f"hist_failures_new_model_batch{batch}.png",
                     title="Histogram of Failures", xlabel="Probability (%)", percentage=True)
    create_histogram(times_us, bins=30, output_path= f"hist_times_new_model_batch{batch}.png",
                     title="Histogram of Inference Times", xlabel="Time (µs)", percentage=False)

    print("Number of success: ", len(correct))
    print("Number of faild: ", len(incorrect))
    print("Avg time: ", avg_time_us, "microseconds")

    # ⚠️ הדפסת זמן כולל
    print("Total inference time: {:.2f} seconds ({:.2f} minutes)".format(total_time_sec, total_time_min))

if __name__ == "__main__":
    print("batch = 1:\n")
    main(1)
    print("batch = 32:\n")
    main(32)
    print("batch = 64:\n")
    main(64)
    print("batch = 128:\n")
    main(128)
    print("batch = 256:\n")
    main(256)
    print("batch = 512:\n")
    main(512)