import unittest
from src.recommend import hybrid_recommendation

class TestRecommendation(unittest.TestCase):
    def test_recommendation(self):
        user_id = 1
        movie_title = "Toy Story (1995)"
        recommendations = hybrid_recommendation(user_id, movie_title)
        self.assertGreater(len(recommendations), 0)

if __name__ == "__main__":
    unittest.main()