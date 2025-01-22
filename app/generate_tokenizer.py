import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import tensorflow as tf
import pickle
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Forcer l'utilisation du CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Réduire les logs inutiles



# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize NLTK tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()



def clean_text(text):
    """
    Cleans input text by removing URLs, mentions, hashtags, non-alphabetic characters,
    and by applying stemming and lemmatization.

    Args:
        text (str): The text to clean.

    Returns:
        list: A list of cleaned and tokenized words.
    """
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



def load_dataframe(file_path):
    """
    Loads a DataFrame from a CSV file, ensuring proper handling of mixed data types.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    try:
        df = pd.read_csv(file_path, low_memory=False)
        if 'text' not in df.columns:
            raise ValueError("The 'text' column is missing in the provided file.")
        if not df['text'].apply(lambda x: isinstance(x, str)).all():
            raise ValueError("Some values in the 'text' column are not strings.")
        return df
    except Exception as e:
        raise ValueError(f"Error loading DataFrame: {e}")



def save_tokenizer(tokenizer, file_path):
    """
    Saves the tokenizer to a pickle file.

    Args:
        tokenizer (tf.keras.preprocessing.text.Tokenizer): The tokenizer to save.
        file_path (str): The path to the pickle file.
    """
    with open(file_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)



def main():
    """
    Main function to load data, clean text, fit a tokenizer, and save the tokenizer.
    """
    try:
        df = load_dataframe('Dataset_Init.csv')
        texts_for_training = df['text'].apply(clean_text).tolist()

        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000, oov_token="<OOV>")
        tokenizer.fit_on_texts(texts_for_training)

        save_tokenizer(tokenizer, 'tokenizer.pickle')
    except Exception as e:
        print(f"Error in main process: {e}")


if __name__ == "__main__":
    main()
