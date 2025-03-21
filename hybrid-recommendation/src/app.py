from flask import Flask, request, jsonify
from src.recommend import hybrid_recommendation

app = Flask(__name__)

@app.route("/recommend", methods=["GET"])
def recommend():
    user_id = int(request.args.get("user_id"))
    movie_title = request.args.get("movie_title")
    recommendations = hybrid_recommendation(user_id, movie_title)
    return jsonify(recommendations.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)