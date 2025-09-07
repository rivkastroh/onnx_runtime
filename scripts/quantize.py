import gzip
import struct
import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    quantize_static,
    QuantType,
    QuantFormat,
    CalibrationMethod
)
import onnx

# ---------- שלב 1: פונקציות לטעינת ubyte.gz ----------
def load_images(path):
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        # magic = 2051 תמיד עבור images
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num, 1, rows, cols)  # [N,1,28,28]
        return data

def load_labels(path):
    with gzip.open(path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        # magic = 2049 תמיד עבור labels
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data

# ---------- שלב 2: DataReader לקליברציה ----------
class MnistUbyteDataReader(CalibrationDataReader):
    def __init__(self, model_path, images, limit=None, batch_size=1):
        self.model = onnx.load(model_path)
        inp = self.model.graph.input[0]
        self.input_name = inp.name

        if limit is not None:
            images = images[:limit]
        self.images = images.astype(np.float32) / 255.0  # נירמול ל-[0,1]
        self.batch_size = batch_size
        self._reset_iter()

    def _reset_iter(self):
        def gen():
            n = self.images.shape[0]
            for i in range(0, n, self.batch_size):
                yield {self.input_name: self.images[i:i+self.batch_size]}
        self.enum_data = gen()

    def get_next(self):
        try:
            return next(self.enum_data)
        except StopIteration:
            return None

    def rewind(self):
        self._reset_iter()

# ---------- שלב 3: טעינת דאטה ----------
train_images = load_images("train-images-idx3-ubyte.gz")
train_labels = load_labels("train-labels-idx1-ubyte.gz")
print("Train images:", train_images.shape)  # (60000, 1, 28, 28)
print("Train labels:", train_labels.shape)  # (60000,)

# ---------- שלב 4: קוונטיזציה ----------
model_fp32 = "mnist_opset13.onnx"
model_int8 = "mnist_opset13.static.int8.onnx"

# נשתמש ב-1000 דגימות ראשונות לקליברציה
dr = MnistUbyteDataReader(model_fp32, train_images, limit=1000, batch_size=1)

quantize_static(
    model_input=model_fp32,
    model_output=model_int8,
    calibration_data_reader=dr,
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    calibrate_method=CalibrationMethod.MinMax,
    per_channel=True,
    reduce_range=False
)

print("Quantized model saved to:", model_int8)
