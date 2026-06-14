import matplotlib.pyplot as plt


def plot_rating_distribution(ratings):
    counts = ratings["rating"].value_counts().sort_index()

    plt.figure(figsize=(7, 4))
    plt.bar(counts.index, counts.values)
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.title("Rating Distribution")
    plt.grid(axis="y", alpha=0.3)
    plt.show()


def plot_interactions_per_user(positive_interactions):
    counts = positive_interactions.groupby("user_id").size()

    plt.figure(figsize=(7, 4))
    plt.hist(counts, bins=30)
    plt.xlabel("Positive interactions per user")
    plt.ylabel("Number of users")
    plt.title("Interactions per User")
    plt.grid(axis="y", alpha=0.3)
    plt.show()


def plot_interactions_per_movie(positive_interactions):
    counts = positive_interactions.groupby("movie_id").size()

    plt.figure(figsize=(7, 4))
    plt.hist(counts, bins=30)
    plt.xlabel("Positive interactions per movie")
    plt.ylabel("Number of movies")
    plt.title("Interactions per Movie")
    plt.grid(axis="y", alpha=0.3)
    plt.show()


def plot_loss_curve(loss_history, title="Training Loss"):
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.show()
