import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def negative_sampling(train_data, positive_interactions, movies, random_state=42):
    all_movie_ids = set(movies["movie_id"].unique())
    user_positive_movies = (
        positive_interactions.groupby("user_id")["movie_id"].apply(set).to_dict()
    )

    rng = np.random.default_rng(random_state)
    negative_rows = []

    for row in train_data.itertuples(index=False):
        candidate_movies = list(all_movie_ids - user_positive_movies[row.user_id])
        negative_movie_id = int(rng.choice(candidate_movies))

        negative_rows.append(
            {
                "user_id": row.user_id,
                "movie_id": negative_movie_id,
                "interaction": 0,
                "timestamp": row.timestamp,
            }
        )

    negative_samples = pd.DataFrame(negative_rows)
    positive_samples = train_data[["user_id", "movie_id", "interaction", "timestamp"]]

    return (
        pd.concat([positive_samples, negative_samples], ignore_index=True)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )


def create_train_loader(train_data, batch_size=1024):
    dataset = TensorDataset(
        torch.tensor(train_data["user_idx"].values, dtype=torch.long),
        torch.tensor(train_data["movie_idx"].values, dtype=torch.long),
        torch.tensor(train_data["interaction"].values, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_with_sgd(model, train_loader, device, lr=0.1, num_epochs=20):
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_history = []

    start_time = time.time()

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_examples = 0

        for batch_users, batch_movies, batch_labels in train_loader:
            batch_users = batch_users.to(device)
            batch_movies = batch_movies.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            scores = model(batch_users, batch_movies)
            loss = loss_function(scores, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_labels)
            total_examples += len(batch_labels)

        loss_history.append(total_loss / total_examples)

    training_time = time.time() - start_time
    return loss_history, training_time
