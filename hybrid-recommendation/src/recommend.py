import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.train import train_model

def hybrid_recommendation(user_id, movie_title):
    movies = pd.read_csv("../data/movies.csv")
    ratings = pd.read_csv("../data/ratings.csv")
    model = train_model()

    # Контентная фильтрация
    tfidf = TfidfVectorizer(stop_words="english")
    movies['genres'] = movies['genres'].fillna("")
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]

    # Коллаборативная фильтрация
    movie_ids = movies.iloc[movie_indices]['movieId']
    predictions = [model.predict(user_id, movie_id).est for movie_id in movie_ids]

    # Объединение результатов
    recommendations = pd.DataFrame({
        'movieId': movie_ids,
        'title': movies.iloc[movie_indices]['title'],
        'predicted_rating': predictions
    }).sort_values('predicted_rating', ascending=False)

    return recommendations

if __name__ == "__main__":
    user_id = 1
    movie_title = "Toy Story (1995)"
    recommendations = hybrid_recommendation(user_id, movie_title)
    print("Hybrid Recommendations:")
    print(recommendations)