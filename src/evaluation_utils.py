import math

import numpy as np
import torch


def recommend_top_k(model, user_idx, num_movies, seen_movies, device, k=10):
    model.eval()

    with torch.no_grad():
        user_tensor = torch.full((num_movies,), user_idx, dtype=torch.long, device=device)
        movie_tensor = torch.arange(num_movies, dtype=torch.long, device=device)
        scores = model(user_tensor, movie_tensor).detach().cpu().numpy()

    scores[list(seen_movies)] = -np.inf
    return np.argsort(scores)[-k:][::-1].tolist()


def hit_rate_at_k(recommended_items, test_item):
    return int(test_item in recommended_items)


def precision_at_k(recommended_items, test_item, k=10):
    return hit_rate_at_k(recommended_items, test_item) / k


def recall_at_k(recommended_items, test_item):
    return hit_rate_at_k(recommended_items, test_item)


def ndcg_at_k(recommended_items, test_item):
    if test_item not in recommended_items:
        return 0.0

    rank = recommended_items.index(test_item) + 1
    return 1 / math.log2(rank + 1)


def evaluate_leave_one_out(model, test_data, train_data, num_movies, device, k=10):
    seen_movies_by_user = train_data.groupby("user_idx")["movie_idx"].apply(set).to_dict()

    precision_scores = []
    recall_scores = []
    hit_rate_scores = []
    ndcg_scores = []

    for row in test_data.itertuples(index=False):
        seen_movies = seen_movies_by_user.get(row.user_idx, set())
        recommendations = recommend_top_k(
            model=model,
            user_idx=row.user_idx,
            num_movies=num_movies,
            seen_movies=seen_movies,
            device=device,
            k=k,
        )

        precision_scores.append(precision_at_k(recommendations, row.movie_idx, k=k))
        recall_scores.append(recall_at_k(recommendations, row.movie_idx))
        hit_rate_scores.append(hit_rate_at_k(recommendations, row.movie_idx))
        ndcg_scores.append(ndcg_at_k(recommendations, row.movie_idx))

    return {
        f"precision@{k}": float(np.mean(precision_scores)),
        f"recall@{k}": float(np.mean(recall_scores)),
        f"hit_rate@{k}": float(np.mean(hit_rate_scores)),
        f"ndcg@{k}": float(np.mean(ndcg_scores)),
    }
