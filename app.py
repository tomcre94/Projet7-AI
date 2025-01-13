import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Charger le modèle LSTM
model = tf.keras.models.load_model('model_lstm.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = np.array(data['input']).reshape(1, -1)
    prediction = model.predict(input_data)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)