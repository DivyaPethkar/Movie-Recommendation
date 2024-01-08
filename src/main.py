import pandas as pd
from surprise import Reader, Dataset
from src.utils import load_data, create_ratings_matrix, split_data
from src.recommender import user_based_cf, item_based_cf, matrix_factorization
from src.evaluation import evaluate_predictions

# Load and preprocess data
ratings_data, movies_data = load_data("data/ratings.csv", "data/movies.csv")
ratings_matrix = create_ratings_matrix(ratings_data)

# Split data into training and testing sets
trainset, testset = split_data(ratings_matrix, test_size=0.2)

# Choose a recommendation algorithm
algorithm = "matrix_factorization"  # or "user_based_cf" or "item_based_cf"

# Train and evaluate the model
if algorithm == "user_based_cf" or algorithm == "item_based_cf":
    recommend_movies = eval(algorithm)(ratings_matrix)  # Call the appropriate function
else:
    model = matrix_factorization(trainset, testset)

# Get recommendations for a user
user_id = 10  # Replace with the desired user ID
num_recommendations = 10

if algorithm == "matrix_factorization":
    predictions = model.test(testset)
    evaluation = evaluate_predictions(predictions, [r[2] for r in testset])
    print(evaluation)

    # Get top N recommendations for a specific user
    user_predictions = [p for p in predictions if p[0] == user_id]
    recommended_movies = [p[1] for p in user_predictions]
else:
    recommended_movies = recommend_movies(user_id, num_recommendations)

print(f"Recommended movies for user {user_id}:")
for movie_id in recommended_movies:
    print(movies_data.loc[movies_data['movieId'] == movie_id, 'title'].iloc[0])
