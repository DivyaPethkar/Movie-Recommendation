import pandas as pd

def load_data(ratings_file, movies_file):
    """
    Loads and preprocesses the ratings and movies data.

    Args:
        ratings_file (str): Path to the ratings CSV file.
        movies_file (str): Path to the movies CSV file.

    Returns:
        tuple: A tuple containing the ratings DataFrame and movies DataFrame.
    """

    ratings_data = pd.read_csv(ratings_file)
    movies_data = pd.read_csv(movies_file)

    # Handle missing values (if any)
    ratings_data.dropna(inplace=True)
    movies_data.dropna(inplace=True)

    # Normalize ratings (optional)
    ratings_data['rating'] = ratings_data['rating'] / 5.0

    return ratings_data, movies_data

def create_ratings_matrix(ratings_data):
    """
    Creates a user-item ratings matrix from the ratings DataFrame.

    Args:
        ratings_data (pandas.DataFrame): The ratings DataFrame.

    Returns:
        numpy.ndarray: The user-item ratings matrix.
    """

    num_users = ratings_data['userId'].max() + 1
    num_movies = ratings_data['movieId'].max() + 1
    ratings_matrix = np.zeros((num_users, num_movies))

    for _, row in ratings_data.iterrows():
        ratings_matrix[int(row['userId']), int(row['movieId'])] = row['rating']

    return ratings_matrix

def split_data(ratings_matrix, test_size=0.2):
    """
    Splits the ratings matrix into training and testing sets.

    Args:
        ratings_matrix (numpy.ndarray): The user-item ratings matrix.
        test_size (float): The proportion of data to use for testing (default: 0.2).

    Returns:
        tuple: A tuple containing the training and testing sets.
    """

    test_set = np.random.choice(ratings_matrix.nonzero()[0], size=int(test_size * ratings_matrix.nnz), replace=False)
    train_set = ratings_matrix.nonzero()[0]
    train_set = np.delete(train_set, test_set)

    return ratings_matrix[train_set], ratings_matrix[test_set]
