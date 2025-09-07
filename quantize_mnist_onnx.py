import os
import gzip
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

# ==== נתיבים ====
MODEL_PATH = "mnist_emnist_blank_cnn_v1.onnx"
IMG_PATH = "train-images-idx3-ubyte-with-empty.gz"
LBL_PATH = "train-labels-idx1-ubyte-with-empty.gz"
MODEL_QUANT_PATH = "mnist_emnist_blank_cnn_v1_quant.onnx"

# ==== פונקציות עזר לטעינת MNIST המקורי ====
def read_idx_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8)
    magic, num, rows, cols = data[:16].view('>i4')
    images = data[16:].reshape(num, rows, cols)
    return images.astype(np.float32) / 255.0  # Normalized

def read_idx_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8)
    magic, num = data[:8].view('>i4')
    labels = data[8:]
    return labels

# ==== טעינת נתונים ====
images = read_idx_images(IMG_PATH)
labels = read_idx_labels(LBL_PATH)
print(f"Loaded {images.shape[0]} images, shape={images.shape}")
print(f"Loaded {labels.shape[0]} labels")

# ==== זיהוי שם הטנסור של הקלט ====
def get_input_name(onnx_path):
    model = onnx.load(onnx_path)
    return model.graph.input[0].name

input_name = get_input_name(MODEL_PATH)
print(f"Detected ONNX input tensor name: {input_name}")

# ==== הכנת דגימות קלט לדגימה (Calibration) ====
calibration_images = images[:100]  # השתמש ב-100 תמונות ראשונות

class MnistDataReader(CalibrationDataReader):
    def __init__(self, calibration_data):
        # עיצוב ל-(batch, channel, height, width)
        self.batch = np.expand_dims(calibration_data, 1)  # shape: (N, 1, 28, 28)
        self.count = self.batch.shape[0]
        self.idx = 0

    def get_next(self):
        if self.idx >= self.count:
            return None
        # שליחת דגימה אחת בכל קריאה (אפשר להגדיל לבאץ' אם תרצי)
        input_data = self.batch[self.idx:self.idx+1]  # (1, 1, 28, 28)
        self.idx += 1
        return {input_name: input_data}

data_reader = MnistDataReader(calibration_images)

# ==== קוונטיזציה סטטית ====
quantize_static(
    MODEL_PATH,
    MODEL_QUANT_PATH,
    data_reader,
    quant_format=QuantType.QInt8,
    per_channel=True
)
print(f"Quantized model saved to {MODEL_QUANT_PATH}")

# ==== בדיקה: הרצת המודל הקוונטי על תמונה אחת ====
session = ort.InferenceSession(MODEL_QUANT_PATH)
example_input = np.expand_dims(calibration_images[0], axis=(0, 1))  # (1, 1, 28, 28)
outputs = session.run(None, {input_name: example_input})
print("Model output for first calibration image:", outputs)
