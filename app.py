from flask import Flask, request, jsonify
import tensorflow as tf
from predict import predict_cataract
from google.cloud import storage
import os

app = Flask(__name__)
model = tf.keras.models.load_model('model/modelnew.h5')

storage_client = storage.Client()

BUCKET_NAME = 'storage-cataract'

def upload_to_gcs(file, filename, timeout=60):
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(filename)
    blob.upload_from_filename(file, timeout=timeout)
    return None

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = file.filename
        filepath = os.path.join('uploads', filename)
        file.save(filepath)
        result = predict_cataract(model, filepath)

        upload_to_gcs(filepath, filename, timeout=120)

        os.remove(filepath)
        return jsonify(result)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(host='0.0.0.0', port=8080, debug=True)