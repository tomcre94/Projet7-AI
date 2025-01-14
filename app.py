import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
import os
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Télécharger les ressources NLTK nécessaires
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialiser les outils de prétraitement
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Charger le modèle avec plus de détails sur les erreurs
try:
    logger.info("Tentative de chargement du modèle...")
    logger.info(f"Répertoire courant : {os.getcwd()}")
    logger.info(f"Contenu du répertoire : {os.listdir()}")

    if os.path.exists('model_lstm.h5'):
        logger.info("Le fichier model_lstm.h5 existe")
        model = tf.keras.models.load_model('model_lstm.h5')
        logger.info("Modèle chargé avec succès")
    else:
        logger.error("Le fichier model_lstm.h5 n'existe pas dans le répertoire")
        model = None
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
    model = None


# Reste de votre code...
# [Le reste du code reste identique jusqu'à la route predict]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            logger.error("Tentative de prédiction avec un modèle non chargé")
            return jsonify({
                'status': 'error',
                'message': 'Le modèle n\'est pas chargé correctement. Vérifiez les logs pour plus de détails.'
            })

        tweet = request.form['tweet']
        logger.info(f"Tweet reçu : {tweet}")

        processed_tweet = clean_text(tweet)
        processed_tweet = ' '.join(processed_tweet)
        logger.info(f"Tweet traité : {processed_tweet}")

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
        logger.error(f"Erreur lors de la prédiction : {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)