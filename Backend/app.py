from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Muat model TensorFlow Lite
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class names (sesuaikan dengan model Anda)
class_names = ['Breaking', 'Overripe', 'RipeFS', 'RipeSS', 'Underripe'] 

# Fungsi untuk pra-pemrosesan gambar
def preprocess_image(image_data):
    img = Image.open(io.BytesIO(image_data)).convert('RGB')
    img = img.resize((224, 224))  # Sesuaikan dengan ukuran input model
    img_array = np.array(img) / 255.0  # Normalisasi
    return np.expand_dims(img_array, axis=0).astype(np.float32)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image_data = image_file.read()

    input_data = preprocess_image(image_data)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Interpretasi output
    predicted_class_id = np.argmax(output_data)
    predicted_class = class_names[predicted_class_id]
    confidence = output_data[0][predicted_class_id]

    response = {
        'class': predicted_class,
        'confidence': float(confidence)
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
