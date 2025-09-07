"""
MNIST Evaluation with ONNX Runtime
---------------------------------
סקריפט זה טוען מודל ONNX מוכן (mnist-12.onnx),
מריץ אותו על סט הבדיקה של MNIST (10,000 ספרות),
ומחשב דיוק ולוס (Cross Entropy).
"""

import numpy as np
import onnxruntime as ort
import gzip
import struct

# ----------------------------------------------------------
# קריאת קובץ תמונות MNIST בפורמט IDX
# ----------------------------------------------------------
def load_images(filename: str) -> np.ndarray:
    """
    קורא קובץ תמונות MNIST דחוס (gz).
    מחזיר מערך בצורה [num_images, 1, 28, 28] מנורמל ל-0..1
    """
    with gzip.open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        # נרמול ל-0..1 והוספת ערוץ יחיד (grayscale)
        return data.reshape(num, 1, rows, cols).astype(np.float32) / 255.0

# ----------------------------------------------------------
# קריאת קובץ לייבלים MNIST בפורמט IDX
# ----------------------------------------------------------
def load_labels(filename: str) -> np.ndarray:
    """
    קורא קובץ לייבלים MNIST דחוס (gz).
    מחזיר מערך של מספרים (0..9) בגודל num_images
    """
    with gzip.open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

# ----------------------------------------------------------
# חישוב Cross Entropy Loss
# ----------------------------------------------------------
def cross_entropy(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    מחשב Cross Entropy Loss בין הסתברויות שהמודל פלט ללייבלים האמיתיים.
    :param probs: [N,10] הסתברויות לכל ספרה
    :param labels: [N] ספרה אמיתית
    """
    eps = 1e-10  # כדי למנוע log(0)
    log_likelihood = -np.log(probs[np.arange(len(labels)), labels] + eps)
    return log_likelihood.mean()

# ----------------------------------------------------------
# שלב ראשי
# ----------------------------------------------------------
if __name__ == "__main__":
    # טעינת סט הבדיקה
    images = load_images("t10k-images-idx3-ubyte.gz")
    labels = load_labels("t10k-labels-idx1-ubyte.gz")

    # טעינת מודל ONNX
    session = ort.InferenceSession("mnist.onnx")

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # הרצת המודל - תמונה אחת בכל פעם
    probs = []
    for i in range(len(images)):
        img = images[i:i+1]  # batch = 1
        outputs = session.run([output_name], {input_name: img})
        probs.append(outputs[0][0])

    probs = np.array(probs)

    # חישוב תחזיות
    preds = np.argmax(probs, axis=1)

    # דיוק
    accuracy = (preds == labels).mean()

    # לוס
    loss = cross_entropy(probs, labels)

    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Cross Entropy Loss: {loss:.4f}")

