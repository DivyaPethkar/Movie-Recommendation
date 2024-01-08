import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, SVD

def user_based_cf(ratings_matrix):
    """
    User-based collaborative filtering.

    Args:
        ratings_matrix (numpy.ndarray): User-item ratings matrix.

    Returns:
        function: A function that takes a user ID and number of recommendations
                 and returns a list of recommended movie IDs.
    """

    user_similarities = cosine_similarity(ratings_matrix)

    def recommend_movies(user_id, num_recommendations):
        user_ratings = ratings_matrix[user_id, :]
        similar_users = np.argsort(-user_similarities[user_id, :])
        recommendations = []
        for similar_user_id in similar_users:
            if similar_user_id != user_id:
                for movie_id in np.where(ratings_matrix[similar_user_id, :] > 0)[0]:
                    if movie_id not in recommendations and user_ratings[movie_id] == 0:
                        recommendations.append(movie_id)
                        if len(recommendations) == num_recommendations:
                            return recommendations
        return recommendations

    return recommend_movies

def item_based_cf(ratings_matrix):
    """
    Item-based collaborative filtering.

    Args:
        ratings_matrix (numpy.ndarray): User-item ratings matrix.

    Returns:
        function: A function that takes a user ID and number of recommendations
                 and returns a list of recommended movie IDs.
    """

    item_similarities = cosine_similarity(ratings_matrix.T)

    def recommend_movies(user_id, num_recommendations):
        user_ratings = ratings_matrix[user_id, :]
        recommended_movies = []
        for movie_id in np.where(user_ratings > 0)[0]:
            similar_movies = np.argsort(-item_similarities[movie_id, :])
            for similar_movie_id in similar_movies:
                if similar_movie_id != movie_id and similar_movie_id not in recommended_movies:
                    recommended_movies.append(similar_movie_id)
                    if len(recommended_movies) == num_recommendations:
                        return recommended_movies
        return recommended_movies

    return recommend_movies

def matrix_factorization(trainset, testset):
    """
    Matrix factorization using SVD algorithm.

    Args:
        trainset (surprise.Trainset): Training set.
        testset (surprise.Trainset): Test set.

    Returns:
        surprise.prediction_algorithms.algo_base.AlgoBase: Trained SVD model.
    """

    algo = SVD()
    algo.fit(trainset)
    predictions = algo.test(testset)

    return algo
