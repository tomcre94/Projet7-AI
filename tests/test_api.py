import unittest
import json
from app.main import app  # Import de votre application Flask

class FlaskAPITestCase(unittest.TestCase):
    def setUp(self):
        # Crée un client de test pour simuler des requêtes HTTP
        self.app = app.test_client()
        self.app.testing = True
        # URL de test pour l'API
        self.api_url = "https://projet7-deeplearning-e9hafvfabugpe0c3.francecentral-01.azurewebsites.net/predict"

    def test_home_route(self):
        # Test de la route d'accueil (GET /)
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.data.decode('utf-8'),
            "API d'analyse de sentiments en ligne"
        )

    def test_predict_route(self):
        # Test de la route de prédiction (POST /predict)
        payload = {'text': 'Je suis très content aujourd\'hui!'}
        response = self.app.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        # Vérifie que la réponse est OK
        self.assertEqual(response.status_code, 200)
        
        # Vérifie que la réponse contient tous les champs attendus
        data = json.loads(response.data)
        self.assertIn('text', data)
        self.assertIn('sentiment', data)
        self.assertIn('score', data)
        self.assertIn('processed_text', data)
        
        # Vérifie que le sentiment est soit positif soit négatif
        self.assertIn(data['sentiment'], ['positif', 'négatif'])
        
        # Vérifie que le score est un nombre entre 0 et 1
        self.assertIsInstance(data['score'], float)
        self.assertTrue(0 <= data['score'] <= 1)

    def test_predict_route_empty_text(self):
        # Test avec un texte vide
        payload = {'text': ''}
        response = self.app.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)

    def test_predict_route_missing_text(self):
        # Test sans le champ 'text'
        payload = {}
        response = self.app.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)

    def test_feedbackpositif_route(self):
        # Test de la route du feedback positif
        response = self.app.post('/feedbackpositif')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.decode('utf-8'), "true")

    def test_feedbacknegatif_route(self):
        # Test de la route du feedback négatif
        payload = {'text': 'Je suis triste'}
        response = self.app.post(
            '/feedbacknegatif',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'Feedback enregistré')


if __name__ == '__main__':
    unittest.main()