import onnx

# טוענים את המודל המקורי
model = onnx.load("mnist_emnist_blank_cnn_v1_quant.onnx")

# הקלט הראשון (בד"כ יחיד) נמצא בגרף
input_tensor = model.graph.input[0]

# מחליפים את הממד הראשון מ-1 ל-N (כלומר דינמי)
input_tensor.type.tensor_type.shape.dim[0].dim_param = "N"

# שומרים עותק חדש
onnx.save(model, "mnist_emnist_blank_cnn_v1_quant_dynamic.onnx")

print("Saved dynamic-batch model as mnist_dynamic.onnx")
