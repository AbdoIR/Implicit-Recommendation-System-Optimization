from pathlib import Path
import pickle

import pandas as pd


MOVIE_COLUMNS = [
    "movie_id", "title", "release_date", "video_release_date", "imdb_url",
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def ensure_project_dirs(root_dir="."):
    root_dir = Path(root_dir)
    paths = [
        "data/processed",
        "models",
        "results",
        "results/figures",
        "results/tables",
    ]

    for path in paths:
        (root_dir / path).mkdir(parents=True, exist_ok=True)


def load_movielens_100k(data_dir="dataset/ml-100k"):
    data_dir = Path(data_dir)

    ratings = pd.read_csv(
        data_dir / "u.data",
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )

    users = pd.read_csv(
        data_dir / "u.user",
        sep="|",
        names=["user_id", "age", "gender", "occupation", "zip_code"],
    )

    movies = pd.read_csv(
        data_dir / "u.item",
        sep="|",
        encoding="latin-1",
        names=MOVIE_COLUMNS,
    )

    return ratings, users, movies


def create_implicit_feedback(ratings, threshold=4):
    ratings = ratings.copy()
    ratings["interaction"] = (ratings["rating"] >= threshold).astype(int)
    positive_interactions = ratings[ratings["interaction"] == 1].copy()
    return ratings, positive_interactions


def compute_sparsity(num_positive_interactions, num_users, num_movies):
    return 1 - (num_positive_interactions / (num_users * num_movies))


def leave_one_out_split(positive_interactions):
    sorted_interactions = positive_interactions.sort_values(["user_id", "timestamp"])
    test_data = sorted_interactions.groupby("user_id").tail(1).copy()
    train_data = sorted_interactions.drop(test_data.index).copy()
    return train_data, test_data


def create_mappings(users, movies):
    user_ids = sorted(users["user_id"].unique())
    movie_ids = sorted(movies["movie_id"].unique())

    user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
    idx_to_user = {idx: user_id for user_id, idx in user_to_idx.items()}
    idx_to_movie = {idx: movie_id for movie_id, idx in movie_to_idx.items()}

    return user_to_idx, movie_to_idx, idx_to_user, idx_to_movie


def add_index_columns(data, user_to_idx, movie_to_idx):
    data = data.copy()
    data["user_idx"] = data["user_id"].map(user_to_idx)
    data["movie_idx"] = data["movie_id"].map(movie_to_idx)
    return data


def save_processed_data(
    train_data,
    test_data,
    movies,
    users,
    metadata,
    output_dir="data/processed",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data.to_csv(output_dir / "train_data.csv", index=False)
    test_data.to_csv(output_dir / "test_data.csv", index=False)
    movies.to_csv(output_dir / "movies_processed.csv", index=False)
    users.to_csv(output_dir / "users_processed.csv", index=False)

    with open(output_dir / "metadata.pkl", "wb") as file:
        pickle.dump(metadata, file)


def load_processed_metadata(path="data/processed/metadata.pkl"):
    with open(path, "rb") as file:
        return pickle.load(file)
