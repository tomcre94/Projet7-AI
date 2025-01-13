import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
import os
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk

app = Flask(__name__)

# Télécharger les ressources NLTK nécessaires
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialiser les outils de prétraitement
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Charger le modèle LSTM
model = tf.keras.models.load_model('model_lstm.h5')


def clean_text(text):
    # Remplacer les URLs par le mot "URL"
    text = re.sub(r'http\S+|www\S+|https\S+', 'URL', text, flags=re.MULTILINE)
    # Remplacer les mentions par le mot "mention"
    text = re.sub(r'\@\w+', 'mention', text)
    # Remplacer les hashtags par le mot "hashtag"
    text = re.sub(r'\#\w+', 'hashtag', text)
    # Suppression des caractères spéciaux et des chiffres
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Conversion en minuscules
    text = text.lower()
    # Tokenisation
    tokens = word_tokenize(text)
    # Suppression des stopwords et de la ponctuation
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    # Stemming et lemmatization
    tokens = [stemmer.stem(word) for word in tokens]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer le tweet depuis le formulaire
        tweet = request.form['tweet']

        # Prétraiter le tweet
        processed_tweet = clean_text(tweet)

        # Convertir en chaîne pour la prédiction
        processed_tweet = ' '.join(processed_tweet)

        # Faire la prédiction
        # Note: Assurez-vous que votre modèle attend les données dans ce format
        input_data = np.array([processed_tweet])
        prediction = model.predict(input_data)

        # Interpréter la prédiction (1 pour positif, 0 pour négatif)
        sentiment = "Positif" if prediction[0] > 0.5 else "Négatif"
        confidence = float(prediction[0]) if prediction[0] > 0.5 else float(1 - prediction[0])

        return jsonify({
            'status': 'success',
            'sentiment': sentiment,
            'confidence': confidence,
            'tweet': tweet,
            'processed_tweet': processed_tweet  # Ajouté pour le débogage
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)