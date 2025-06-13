from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load model
MODEL_PATH = 'meowid_model.h5'
model = load_model(MODEL_PATH)
print("MODEL INPUT SHAPE:", model.input_shape)

# Label sesuai urutan pelatihan
class_names = ['Anggora', 'Bengal', 'Persian', 'Siamese', 'Sphynx', 'Tabby']

@app.route('/')
def index():
    return "MeowID Backend is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Resize sesuai input model: 150x150
        img = Image.open(file.stream).convert("RGB").resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 150, 150, 3)

        prediction = model.predict(img_array)[0]
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction)) * 100

        return jsonify({
            'class': predicted_class,
            'confidence': f'{confidence:.2f}%'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
