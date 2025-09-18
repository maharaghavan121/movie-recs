import numpy as np
import pandas as pd
import scipy.sparse as sp


def load_data():
    movies = pd.read_csv("data_small/movies.csv")
    ratings = pd.read_csv("data_small/ratings.csv")
    links = pd.read_csv("data_small/links.csv")
    tags = pd.read_csv("data_small/tags.csv")
    return movies, ratings, links, tags


def make_splits(ratings, thresh = 4):
    df = ratings.copy()
    df = df.sort_values(['userId', 'timestamp'])
    df['is_pos'] = df["rating"] >= thresh

    pos = df[df["is_pos"]].copy()
    # Rank positives within each user (1..n)
    pos["rnk"] = pos.groupby("userId")["timestamp"].rank(method = "first")

    # num_of_positives per user 
    n_pos = pos.groupby("userId")["rnk"].max().rename("n_pos")
    pos = pos.merge(n_pos, on = "userId")

    # keep only users with at least 2 positives
    pos = pos[pos["n_pos"] >= 2].copy()

    # create test, val, train splits
    test = pos[pos["rnk"] == pos["n_pos"]].copy()
    val = pos[pos["rnk"] == pos["n_pos"] - 1].copy()
    train = pos[pos["rnk"] < pos["n_pos"] - 1].copy()
    
    return train, val, test

def build_index_maps(ratings, train_data):
    users = np.sort(train_data["userId"].unique())
    items = np.sort(train_data["movieId"].unique())
    # Lookup take from id to index
    user_idx = {u: i for i, u in enumerate(users)}
    item_idx = {i: i for i, m in enumerate(items)}
    return users, items, user_idx, item_idx

def positives_to_csr(train_data, users, items, user_idx, item_idx, alpha = 20):
    rows = train_data["movieId"].map(item_idx).values
    cols = train_data["userId"].map(user_idx).values
    data = np.full(len(train_data), 1.0 + alpha, dtype = "float32")
    # X is a sparse coordinate matrix that is built from triplets. 
    # tocsr is a sparse matrix in Compressed Sparse Row format
    X = sp.coo_matrix((data, (rows, cols)), shape = (len(items), len(users))).tocsr()
    return X


def main():
    print("Hello from movie-recs!")
    # Load the data -
    print("Loading the data")
    movies, ratings, links, tags = load_data()

    # Split the data for training and testing
    print("Splitting the data for training and testing")
    train_data, val_data, test_data = make_splits(ratings, thresh = 4)

    print("Indexing and building sparse train matrix for ALS")
    users, items, user_idx, item_idx = build_index_maps(ratings, train_data)
    x_train = positives_to_csr(train_data, users, items, user_idx, item_idx, alpha = 20)

if __name__ == "__main__":
    main()
