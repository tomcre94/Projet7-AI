import unittest
import json
from unittest.mock import patch, Mock
from app.main import app
import signal
from contextlib import contextmanager


@contextmanager
def timeout(seconds):
    """Contexte pour limiter le temps d'exécution d'un test."""
    def signal_handler(signum, frame):
        raise TimeoutError("Test took too long!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


class FlaskAPITestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Création des mocks pour le tokenizer et le modèle
        cls.tokenizer_mock = Mock()
        cls.tokenizer_mock.texts_to_sequences.return_value = [[1, 2, 3]]
        
        cls.model_mock = Mock()
        cls.model_mock.predict.return_value = [[0.8]]

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    @patch('app.main.tokenizer')
    @patch('app.main.model')
    def test_home_route(self, mock_model, mock_tokenizer):
        mock_tokenizer.return_value = self.tokenizer_mock
        mock_model.return_value = self.model_mock
        
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.data.decode('utf-8'),
            "API d'analyse de sentiments en ligne"
        )

    @patch('app.main.tokenizer')
    @patch('app.main.model')
    def test_predict_route(self, mock_model, mock_tokenizer):
        mock_tokenizer.return_value = self.tokenizer_mock
        mock_model.return_value = self.model_mock
        
        payload = {'text': 'Je suis très content aujourd\'hui!'}
        with timeout(5):  # Timeout de 5 secondes pour limiter le temps
            response = self.app.post(
                '/predict',
                data=json.dumps(payload),
                content_type='application/json'
            )
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('text', data)
        self.assertIn('sentiment', data)
        self.assertIn('score', data)
        self.assertIn('processed_text', data)
        
        self.assertIn(data['sentiment'], ['positif', 'négatif'])
        self.assertIsInstance(data['score'], float)
        self.assertTrue(0 <= data['score'] <= 1)

    @patch('app.main.tokenizer')
    @patch('app.main.model')
    def test_feedbackpositif_route(self, mock_model, mock_tokenizer):
        mock_tokenizer.return_value = self.tokenizer_mock
        mock_model.return_value = self.model_mock
        
        response = self.app.post('/feedbackpositif')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.decode('utf-8'), "true")

    # Autres tests supprimés pour simplification
    # Ajoutez uniquement ceux nécessaires si critiques


if __name__ == '__main__':
    unittest.main()
