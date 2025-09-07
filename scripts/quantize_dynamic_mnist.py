from onnxruntime.quantization import quantize_dynamic, QuantType

# קובץ המקור (מודל FP32)
model_fp32 = "mnist_emnist_blank_cnn_v1.onnx"

# שם הקובץ החדש (מודל INT8)
model_int8 = "mnist_emnist_blank_cnn_v1.int8.onnx"

# הפעלת קוונטיזציה דינמית
quantize_dynamic(
    model_input=model_fp32,       # קובץ המקור
    model_output=model_int8,      # קובץ היעד
    weight_type=QuantType.QUInt8,  # המרת משקלים ל־INT8
    per_channel=True              # שימוש ב־per-channel (מדויק יותר)
)

print("✔ קוונטיזציה הושלמה בהצלחה!")
print(f"נוצר קובץ חדש: {model_int8}")
