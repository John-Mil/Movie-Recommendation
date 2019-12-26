"""
John Milmore
==============================================
Compare KNN Collaborative Filtering Algorithms
==============================================
"""

from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

import matrix_cf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

URL = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
RESP = urlopen(URL)
ZIPFILE = ZipFile(BytesIO(RESP.read()))


def get_rating_data(rating_file='ml-1m/ratings.dat', max_rows=1e6):
    """Return rating data extracted online from rating_file."""
    userID_ls = []
    movieID_ls = []
    rating_ls = []

    for i, line in enumerate(ZIPFILE.open(rating_file).readlines()):
        if i >= max_rows:
            break
        try:
            x = line.decode('utf-8').split('::')
        except Exception:
            continue
        userID_ls.append(int(x[0]))
        movieID_ls.append(int(x[1]))
        rating_ls.append(int(x[2]))

    rating_dict = {'userID': np.array(userID_ls),
                   'movieID': np.array(movieID_ls),
                   'rating': np.array(rating_ls)}

    return pd.DataFrame(rating_dict)


def test_splits(ratings_df, predict_method, k, test_size=250, splits=3):
    """Test the prediction MSE of different CF algorithms on small test samples of the rating data.

        Arguments:
            rating_df -- pd DataFrame
                rating data to split into train/test
            predict_method -- function
                KNN CF algorithm
            k -- int
                number of neighbors to consider
            test_size -- int
                number of rows used for evaluation
            cv -- int
                number of folds in cross validation

        Returns:
            average mse of the predicted ratings given from predict_method
    """
    mse_ls = []
    for _ in range(splits):
        test_ratings_df = ratings_df.sample(n=test_size)
        train_ratings_df = ratings_df.drop(test_ratings_df.index)
        model_params = matrix_cf.get_model_params(train_ratings_df)
        _, mse = predict_method(test_ratings_df, k, model_params)
        mse_ls.append(mse)
    return np.array(mse_ls).mean()


def main():
    ##########################################################################
    # Extract data straight from website
    ratings_df = get_rating_data(max_rows=300000)

    # Evaluate users
    val, cts = np.unique(ratings_df['userID'].values, return_counts=True)
    print(f'Average number of ratings per user: {cts.mean():.4f}')
    print(f'Least amount of movies rated by a user: {cts.min()}')
    print(f'Most amount of movies rated by a user: {cts.max()}')
    print(f'Median number of ratings per user: {np.median(cts)}')

    # Evaluate movies
    val, cts = np.unique(ratings_df['movieID'].values, return_counts=True)
    print(f'\nAverage number of ratings per movie: {cts.mean():.4f}')
    print(f'Least amount of ratings for a movie: {cts.min()}')
    print(f'Most amount of rating for a movie: {cts.max()}')
    print(f'Median number of ratings per movie: {np.median(cts)}')

    ##########################################################################
    # Find optimal number of neighbors
    k_ls = [5, 10, 25, 50, 75, 100]
    mse_user_item_ls, mse_item_ls, mse_user_ls = [], [], []
    predict_method_ls = [matrix_cf.predict_user_item, matrix_cf.predict_item, matrix_cf.predict_user]
    mse_ls_ls = [mse_user_item_ls, mse_item_ls, mse_user_ls]
    for k in k_ls:
        for predict_method, mse_ls in zip(predict_method_ls, mse_ls_ls):
            print(f'Testing {k} neighbors')
            mse = test_splits(ratings_df, predict_method, k=k)
            mse_ls.append(mse)

    # Find min mse
    min_mse = min(mse_user_item_ls + mse_item_ls + mse_user_ls)
    min_k = None
    for ls in [mse_user_item_ls, mse_item_ls, mse_user_ls]:
        if min_mse in ls:
            idx = np.argmin(ls)
            min_k = k_ls[idx]

    # Plot results
    sns.set()
    fig, ax = plt.subplots(dpi=600)
    ax.plot(k_ls, mse_user_item_ls, '.-', label='User/Item Based')
    ax.plot(k_ls, mse_item_ls, '.-', label='Item Based')
    ax.plot(k_ls, mse_user_ls, '.-', label='User Based')
    ax.set_xlabel('Number of Neighbors')
    ax.set_ylabel('MSE')
    ax.set_title('Testing MSE of Different Memory Based CF Methods', fontweight='bold')
    ax.legend()
    ax.plot(min_k, min_mse, marker='x', c='r', markersize=12)
    fig.savefig('MSE_vs_k_grouplens.png')


if __name__ == '__main__':
    main()
