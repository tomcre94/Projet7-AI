import requests

url = 'http://127.0.0.1:5000/predict'
payload = {'text': 'Je suis très content aujourd\'hui!'}
headers = {'Content-Type': 'application/json'}

response = requests.post(url, json=payload, headers=headers)

print(response.json())