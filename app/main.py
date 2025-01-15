# main.py
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import pickle
import os

app = Flask(__name__)

# Initialisation des outils NLTK
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Charger le modèle
model = tf.keras.models.load_model('model_lstm_compatible.h5')

# Charger le tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


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


@app.route('/')
def home():
    return "API d'analyse de sentiments en ligne"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer le texte de la requête
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Aucun texte fourni'}), 400

        # Prétraitement du texte
        clean_tokens = clean_text(data['text'])
        text_cleaned = ' '.join(clean_tokens)

        # Tokenization avec padding
        sequences = tokenizer.texts_to_sequences([text_cleaned])
        padded_seq = tf.keras.preprocessing.sequence.pad_sequences(
            sequences,
            maxlen=512,
            padding='post',
            truncating='post'
        )

        # Prédiction
        prediction = model.predict(padded_seq)
        sentiment_score = float(prediction[0][0])

        # Classification du sentiment
        sentiment = "positif" if sentiment_score >= 0.5 else "négatif"

        return jsonify({
            'text': data['text'],
            'sentiment': sentiment,
            'score': sentiment_score,
            'processed_text': text_cleaned
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Download required NLTK data
    import nltk

    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)