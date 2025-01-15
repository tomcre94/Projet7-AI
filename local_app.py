import requests

def get_sentiment(tweet):
    url = 'https://projet7-deeplearning.herokuapp.com/predict'
    data = {'tweet': tweet}
    response = requests.post(url, data=data)
    return response.json()

if __name__ == '__main__':
    tweet = input("Entrez votre tweet : ")
    result = get_sentiment(tweet)
    if result['status'] == 'success':
        print(f"Sentiment : {result['sentiment']}")
        print(f"Confiance : {result['confidence'] * 100:.2f}%")
    else:
        print(f"Erreur : {result['message']}")