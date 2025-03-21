import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

def train_model():
    ratings = pd.read_csv("../data/ratings.csv")
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset, _ = train_test_split(data, test_size=0.2)
    model = SVD()
    model.fit(trainset)
    return model

if __name__ == "__main__":
    model = train_model()
    # Сохранение модели можно добавить здесь