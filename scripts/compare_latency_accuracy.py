import time
import numpy as np
import onnxruntime as ort

# קבצי המודלים
model_fp32 = "mnist_emnist_blank_cnn_v1.onnx"
model_int8 = "mnist_emnist_blank_cnn_v1.int8.onnx"

# הכנת סשנים
sess_fp32 = ort.InferenceSession(model_fp32, providers=["CPUExecutionProvider"])
sess_int8 = ort.InferenceSession(model_int8, providers=["CPUExecutionProvider"])

# שמות הקלט
input_name_fp32 = sess_fp32.get_inputs()[0].name
input_name_int8 = sess_int8.get_inputs()[0].name

# פונקציה לחישוב זמן ריצה ממוצע
def benchmark(session, input_name, x, repeat=100):
    # warmup
    for _ in range(10):
        session.run(None, {input_name: x})
    # מדידה
    t0 = time.time()
    for _ in range(repeat):
        session.run(None, {input_name: x})
    t1 = time.time()
    return (t1 - t0) * 1000.0 / repeat  # זמן ממוצע במילישניות

# יצירת דוגמת קלט רנדומלית בגודל מתאים (MNIST: 1x1x28x28)
x = np.random.rand(1, 1, 28, 28).astype(np.float32)

# הרצה והשוואה
lat_fp32 = benchmark(sess_fp32, input_name_fp32, x)
lat_int8 = benchmark(sess_int8, input_name_int8, x)

out_fp32 = sess_fp32.run(None, {input_name_fp32: x})
out_int8 = sess_int8.run(None, {input_name_int8: x})

pred_fp32 = np.argmax(out_fp32[0])
pred_int8 = np.argmax(out_int8[0])

print("=== תוצאות השוואה ===")
print(f"FP32 latency: {lat_fp32:.3f} ms")
print(f"INT8 latency: {lat_int8:.3f} ms")
print(f"Prediction FP32: {pred_fp32}")
print(f"Prediction INT8: {pred_int8}")
