import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

app = Flask(__name__)

# Configuration pour Heroku
if os.environ.get('HEROKU'):
    # Définir le chemin NLTK_DATA
    nltk.data.path.append('./nltk_data/')
    # Télécharger les ressources NLTK si nécessaire
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', download_dir='./nltk_data/')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', download_dir='./nltk_data/')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', download_dir='./nltk_data/')

# Initialiser les outils de prétraitement
try:
    stop_words = set(stopwords.words('english'))
except:
    print("Erreur lors du chargement des stopwords")
    stop_words = set()

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Charger le modèle de manière sécurisée
try:
    model = tf.keras.models.load_model('model_lstm.h5')
    print("Modèle chargé avec succès")
except Exception as e:
    print(f"Erreur lors du chargement du modèle: {str(e)}")
    model = None


def clean_text(text):
    try:
        # Remplacer les URLs
        text = re.sub(r'http\S+|www\S+|https\S+', 'URL', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+', 'mention', text)
        text = re.sub(r'\#\w+', 'hashtag', text)
        text = re.sub(r'[^A-Za-z\s]', '', text)
        text = text.lower()

        # Tokenisation avec gestion d'erreur
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()

        # Filtrage
        tokens = [word for word in tokens if word not in stop_words and word.isalpha()]

        # Stemming et lemmatization
        try:
            tokens = [stemmer.stem(word) for word in tokens]
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
        except Exception as e:
            print(f"Erreur lors du stemming/lemmatization: {str(e)}")

        return tokens
    except Exception as e:
        print(f"Erreur dans clean_text: {str(e)}")
        return text.split()


@app.route('/')
def home():
    return render_template('index.html')


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
    # Utiliser le port défini par Heroku
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)