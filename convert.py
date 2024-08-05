import tensorflow as tf

# Path to the frozen graph .pb file
input_pb = '/content/frozen_inference_graph.pb'  # Replace with the correct path

# Load the model from the frozen graph
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    input_pb,
    input_arrays=['normalized_input_image_tensor'],
    output_arrays=['TFLite_Detection_PostProcess', 'TFLite_Detection_PostProcess:1', 'TFLite_Detection_PostProcess:2', 'TFLite_Detection_PostProcess:3'],  # Use typical SSD MobileNet V3 output names
    input_shapes={'normalized_input_image_tensor': [1, 320, 320, 3]}
)

# Enable custom ops
converter.allow_custom_ops = True

# Convert the model to TensorFlow Lite format
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open('converted_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model successfully converted to TensorFlow Lite format!")
