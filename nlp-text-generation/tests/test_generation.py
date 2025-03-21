import unittest
from src.generate import generate_text

class TestTextGeneration(unittest.TestCase):
    def test_generation(self):
        prompt = "The future of AI is"
        model_path = "../models/gpt2-finetuned"
        result = generate_text(prompt, model_path, max_length=20)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

if __name__ == "__main__":
    unittest.main()