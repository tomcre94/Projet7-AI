import streamlit as st
import requests
import pandas as pd
import json


def analyze_sentiment(text, api_url):
    try:
        response = requests.post(
            api_url,
            json={"text": text},
            timeout=10
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Erreur de connexion: {str(e)}"}


def main():
    st.set_page_config(page_title="Analyseur de Sentiments", page_icon="🎭")

    st.title("🎭 Analyseur de Sentiments de Tweets")

    # Configuration de l'API
    api_url = st.sidebar.text_input(
        "URL de l'API",
        value="https://projet7-deeplearning.azurewebsites.net/predict",
        key="api_url"
    )

    # Zone de texte pour le tweet
    tweet = st.text_area("Entrez votre tweet :", height=100)

    # Bouton d'analyse
    if st.button("Analyser le sentiment"):
        if tweet:
            with st.spinner('Analyse en cours...'):
                result = analyze_sentiment(tweet, api_url)

                if "error" in result:
                    st.error(f"Erreur : {result['error']}")
                else:
                    # Affichage du résultat
                    col1, col2 = st.columns(2)

                    with col1:
                        sentiment = result.get('sentiment', 'inconnu')
                        score = result.get('score', 0)

                        if sentiment == "positif":
                            st.success(f"Sentiment : {sentiment}")
                        else:
                            st.error(f"Sentiment : {sentiment}")

                        st.metric("Score", f"{score:.2%}")

                    with col2:
                        # Feedback
                        st.write("Le résultat est-il correct ?")
                        col_yes, col_no = st.columns(2)

                        with col_yes:
                            if st.button("✅ Oui"):
                                # Sauvegarder le feedback positif
                                feedback = {
                                    'tweet': tweet,
                                    'predicted_sentiment': sentiment,
                                    'score': score,
                                    'feedback': 'correct',
                                    'timestamp': pd.Timestamp.now().isoformat()
                                }
                                st.session_state.setdefault('feedback_data', []).append(feedback)
                                st.success("Merci pour votre feedback !")

                        with col_no:
                            if st.button("❌ Non"):
                                # Sauvegarder le feedback négatif
                                feedback = {
                                    'tweet': tweet,
                                    'predicted_sentiment': sentiment,
                                    'score': score,
                                    'feedback': 'incorrect',
                                    'timestamp': pd.Timestamp.now().isoformat()
                                }
                                st.session_state.setdefault('feedback_data', []).append(feedback)
                                st.error("Merci pour votre feedback. Nous améliorerons notre modèle.")
        else:
            st.warning("Veuillez entrer un tweet à analyser.")

    # Exportation des feedbacks
    if 'feedback_data' in st.session_state and st.session_state.feedback_data:
        if st.button("Télécharger les feedbacks"):
            df_feedback = pd.DataFrame(st.session_state.feedback_data)
            csv = df_feedback.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger CSV",
                data=csv,
                file_name="sentiment_feedback.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()