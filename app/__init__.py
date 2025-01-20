from flask import Flask
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import os

app = Flask(__name__)

# Charger le modèle une seule fois au démarrage
model_path = os.path.join(os.path.dirname(__file__), 'model_lstm_compatible.h5')
model = load_model(model_path)

# Paramètres du modèle
MAX_SEQUENCE_LENGTH = 100  # Ajustez selon votre modèle

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({'error': 'Texte manquant dans la requête'}), 400

        # Prétraitement
        text = data['text']
        sequences = tokenizer.texts_to_sequences([text])
        padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

        # Prédiction
        prediction = model.predict(padded_sequences)
        sentiment_score = float(prediction[0][0])

        # Classification
        if sentiment_score >= 0.7:
            sentiment = "Très positif"
        elif sentiment_score >= 0.5:
            sentiment = "Positif"
        elif sentiment_score >= 0.3:
            sentiment = "Négatif"
        else:
            sentiment = "Très négatif"

        return jsonify({
            'text': text,
            'sentiment': sentiment,
            'score': sentiment_score
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
