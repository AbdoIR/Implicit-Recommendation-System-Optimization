import torch
from torch import nn


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImplicitMatrixFactorization(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=32):
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.movie_embeddings = nn.Embedding(num_movies, embedding_dim)

        nn.init.normal_(self.user_embeddings.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.movie_embeddings.weight, mean=0.0, std=0.01)

    def forward(self, user_idx, movie_idx):
        user_vector = self.user_embeddings(user_idx)
        movie_vector = self.movie_embeddings(movie_idx)
        return (user_vector * movie_vector).sum(dim=1)


def count_model_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())
