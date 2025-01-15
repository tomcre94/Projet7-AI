import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
import os
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk

app = Flask(__name__)

# Télécharger les ressources NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Charger le modèle compatible
try:
    model = tf.keras.models.load_model('model_lstm_compatible.h5')
    print("Modèle chargé avec succès")
except Exception as e:
    print(f"Erreur lors du chargement du modèle: {str(e)}")
    model = None

def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', 'URL', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+', 'mention', text)
    text = re.sub(r'\#\w+', 'hashtag', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    tokens = [stemmer.stem(word) for word in tokens]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({
                'status': 'error',
                'message': 'Le modèle n\'est pas chargé correctement'
            })

        tweet = request.form['tweet']
        processed_tweet = clean_text(tweet)
        processed_tweet = ' '.join(processed_tweet)

        # Faire la prédiction
        input_data = np.array([processed_tweet])
        prediction = model.predict(input_data)

        sentiment = "Positif" if prediction[0] > 0.5 else "Négatif"
        confidence = float(prediction[0]) if prediction[0] > 0.5 else float(1 - prediction[0])

        return jsonify({
            'status': 'success',
            'sentiment': sentiment,
            'confidence': confidence,
            'tweet': tweet,
            'processed_tweet': processed_tweet
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)