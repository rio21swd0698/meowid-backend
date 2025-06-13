from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import requests

app = Flask(__name__)

# === DOWNLOAD MODEL JIKA BELUM ADA ===
MODEL_PATH = 'meowid_model.h5'

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        file_id = '1rp1tjByE4tVFF00kiKGWVvyPPYDjrs61' 
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(url, allow_redirects=True)
        if response.status_code == 200:
            with open(MODEL_PATH, 'wb') as f:
                f.write(response.content)
            print("Model downloaded.")
        else:
            print("Failed to download model:", response.status_code)
            raise Exception("Failed to download model")

download_model()

# === LOAD MODEL ===
model = load_model(MODEL_PATH)
print("MODEL INPUT SHAPE:", model.input_shape)

# === KLASIFIKASI ===
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
        img = Image.open(file.stream).convert("RGB").resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # (1, 150, 150, 3)

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
    CORS(app)
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
