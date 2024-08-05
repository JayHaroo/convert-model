import tensorflow as tf

# Directory of the SavedModel
model_dir = '/content/320'

# Load the SavedModel
loaded_model = tf.saved_model.load(model_dir)

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(model_dir)
tflite_model = converter.convert()

# Save the TFLite model
tflite_model_path = 'model.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f'TFLite model has been saved to {tflite_model_path}')
