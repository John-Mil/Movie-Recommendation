import numpy as np
from sklearn.metrics import mean_squared_error


def get_model_params(rating_df):
    """Transform rating data into (sparse) matrix and gather data necessary for item-based CF.

        Arguments:
            rating_df -- DataFrame storing rating data. cols - 'userID', 'movieID', 'rating'
            replace_mean -- (optional) boolean deciding whether to replace all missing ratings with the average rating
                of the user.

        Returns:
            model_params -- dictionary containing the following
                A - 2D numpy array in which each row represents the vector of ratings for a movie and each column
                    represents a vector of ratings from a user (rows are movies and columns are users). Hence, the
                    matrix has # movies rows and # users columns.
                userID_hash - dictionary (hash table) mapping userID to column index in A.
                movieID_hash - dictionary (hash table) mapping movieID to row index in A.
                user_rated_dict - dictionary mapping hashed values of userID (column index) the list of movies they
                    rated (list of hashed values of movieIDs (row index)).
                avg_user_rating_arr - np array containing the average rating of each user. Index i corresponds to column
                    index of a user.
                movie_rated_by_dict -- dict mapping movie_row index to list of user_cols that have rated movie_row.
                avg_movie_rating_arr -- np array of average movie ratings (length = num of movies).
    """

    def get_hash(arr):
        """Return a hash table for elements in arr."""
        hash_dict = dict()
        ctr = 0
        for el in arr:
            hash_dict[el] = ctr
            ctr += 1
        return hash_dict

    def update_average(avg_ls, idx, rating):
        avg, n = avg_ls[idx]
        curr_sum = avg * n
        new_sum = curr_sum + rating
        new_avg = new_sum / (n + 1)
        avg_ls[idx] = (new_avg, n + 1)

    userID_arr = rating_df['userID'].values
    movieID_arr = rating_df['movieID'].values
    rating_arr = rating_df['rating'].values

    # Unique users and movies
    userID_uniq = np.unique(userID_arr)
    movieID_uniq = np.unique(movieID_arr)

    # Instantiate Matrix
    num_rows = len(movieID_uniq)
    num_cols = len(userID_uniq)
    A = np.zeros((num_rows, num_cols))

    # Hash table for users and movies to matrix indices
    userID_hash = get_hash(userID_uniq)
    movieID_hash = get_hash(movieID_uniq)

    # Movies that each user rated (dense)
    # Maps user_col index to list of movie_rows
    user_rated_dict = {}
    # Users that have rated each movie (dense)
    # Maps movie_row index to list of user_cols
    movie_rated_by_dict = {}

    # Average rating of each user
    # Stores (avg, n) pairs
    avg_user_rating_ls = [(0, 0)] * num_cols
    # Average rating of each movie
    # Stores (avg, m) pairs
    avg_movie_rating_ls = [(0, 0)] * num_rows

    # Fill matrix with ratings values
    # (and store list of movies rated for each user)
    # (and calculate average user rating)
    for userID, movieID, rating in zip(userID_arr, movieID_arr, rating_arr):
        movie_row = movieID_hash[movieID]
        user_col = userID_hash[userID]
        A[movie_row, user_col] = rating
        if user_col in user_rated_dict:
            user_rated_dict[user_col].append(movie_row)
            update_average(avg_user_rating_ls, user_col, rating)
        else:
            user_rated_dict[user_col] = [movie_row]
            avg_user_rating_ls[user_col] = (rating, 1)
        if movie_row in movie_rated_by_dict:
            movie_rated_by_dict[movie_row].append(user_col)
            update_average(avg_movie_rating_ls, movie_row, rating)
        else:
            movie_rated_by_dict[movie_row] = [user_col]
            avg_movie_rating_ls[movie_row] = (rating, 1)

    # Testing
    assert sum(sum(A)) == sum(rating_arr)

    avg_user_rating_arr = np.array([node[0] for node in avg_user_rating_ls])
    avg_movie_rating_arr = np.array([node[0] for node in avg_movie_rating_ls])

    model_params = {'A': A,
                    'userID_hash': userID_hash,
                    'movieID_hash': movieID_hash,
                    'user_rated_dict': user_rated_dict,
                    'avg_user_rating_arr': avg_user_rating_arr,
                    'movie_rated_by_dict': movie_rated_by_dict,
                    'avg_movie_rating_arr': avg_movie_rating_arr}

    return model_params


def predict_user_item(test_rating_df, k, model_params):
    """Returns array of predicted value for each (user, movie) pair in test_rating_df along with MSE. Does so using a
        combination of the Item-Based and User-Based KNN Collaborative Filtering algorithm.

            Arguments:
                test_rating_df -- DataFrame storing labeled rating data to predict. cols - 'userID', 'movieID', 'rating'
                k -- integer representing number of neighbors to consider - number of movies that the user has already rated
                    taken into consideration when predicting the rating of the user on some movie it has not rated.
                model_params -- dictionary containing existing rating data user to calculate a prediction for each user and
                    movie pair.

            Returns:
                pred_rating_arr -- np array consisting of the predictions for each row of (user, movie) in test_rating_df.
                mse -- float representing the mean squared error of the calculated predictions.

    This algorithm calculates the k-nearest movies and k-nearest users for the movie and user of interest. The rating
    prediction is calculated as a weighted average of the ratings of these movies and users, weighted by the adjusted
    cosine similarity score.
    """
    # Predict userID's rating of movieID
    A = model_params['A']  # movie-user matrix
    avg_user_rating_arr = model_params['avg_user_rating_arr']  # average user ratings
    avg_movie_rating_arr = model_params['avg_movie_rating_arr']  # average movie ratings

    userID_arr = test_rating_df['userID'].values
    movieID_arr = test_rating_df['movieID'].values
    rating_arr = test_rating_df['rating'].values  # True ratings
    pred_rating_ls = []

    A_item = A.copy()
    # Replace missing values (unrated movies) with the user's average rating (helpful for similarity)
    for user_idx, col in enumerate(A_item.T):
        col[col == 0] = avg_user_rating_arr[user_idx]

    A_user = A.T.copy()
    # Replace missing values (unrated movies) with the user's average rating (helpful for similarity)
    for movie_idx, row in enumerate(A_user.T):
        row[row == 0] = avg_movie_rating_arr[movie_idx]

    # Make prediction for each (user, movie) pair
    for userID, movieID in zip(userID_arr, movieID_arr):

        try:
            user_col = model_params['userID_hash'][userID]  # column index in A that represents userID
            movie_row = model_params['movieID_hash'][movieID]  # row index in A that represents movieID
            user_rated_ls = model_params['user_rated_dict'][user_col]  # list of movies that userID has rated
            movie_rated_by_ls = model_params['movie_rated_by_dict'][movie_row]  # list of users that rated movieID
        except KeyError as err0:
            print('User/Movie not seen before', err0)
            pred_rating_ls.append(3)  # cheating
            continue

        # Item based
        # Get the k most similar movies that userID has rated
        k_nearest_movies = _knn(movie_row, k, A_item, user_rated_ls, avg_user_rating_arr)

        # User based
        # Get the k most similar users that have rated movieID
        k_nearest_users = _knn(user_col, k, A_user, movie_rated_by_ls, avg_movie_rating_arr)

        if not k_nearest_movies and not k_nearest_users:
            # User hasn't rated any other movies and movie hasn't been rated by any other users
            prediction = 3  # cheating
        else:

            if k_nearest_movies and k_nearest_users:
                num = denom = 0
                for sim, movie_row_i in k_nearest_movies:
                    if np.isnan([sim]):
                        continue
                    num += sim * A[movie_row_i, user_col]
                    denom += abs(sim)
                for sim, user_col_i in k_nearest_users:
                    if np.isnan([sim]):
                        continue
                    num += sim * A[movie_row, user_col_i]
                    denom += abs(sim)
                prediction = num / denom
            elif k_nearest_movies:
                # Just movies
                num = denom = 0
                for sim, movie_row_i in k_nearest_movies:
                    if np.isnan([sim]):
                        continue
                    num += sim * A[movie_row_i, user_col]
                    denom += abs(sim)
                prediction = num / denom
            else:
                # Just users
                num = denom = 0
                for sim, user_col_i in k_nearest_users:
                    if np.isnan([sim]):
                        continue
                    num += sim * A[movie_row, user_col_i]
                    denom += abs(sim)
                prediction = num / denom
            if np.isnan([prediction]):
                print('NaN prediction')
                prediction = 3  # cheating
            pred_rating_ls.append(prediction)

    # Calculate MSE
    pred_rating_arr = np.array(pred_rating_ls)
    mse = mean_squared_error(rating_arr, pred_rating_arr)

    return pred_rating_arr, mse


def predict_item(test_rating_df, k, model_params):
    """Returns array of predicted value for each (user, movie) pair in test_rating_df along with MSE. Does so using the
    Item-Based KNN Collaborative Filtering algorithm.

        Arguments:
            test_rating_df -- DataFrame storing labeled rating data to predict. cols - 'userID', 'movieID', 'rating'
            k -- integer representing number of neighbors to consider - number of movies that the user has already rated
                taken into consideration when predicting the rating of the user on some movie it has not rated.
            model_params -- dictionary containing existing rating data user to calculate a prediction for each user and
                movie pair.

        Returns:
            pred_rating_arr -- np array consisting of the predictions for each row of (user, movie) in test_rating_df.
            mse -- float representing the mean squared error of the calculated predictions.
    """
    # Predict userID's rating of movieID
    A = model_params['A']  # movie-user matrix
    avg_user_rating_arr = model_params['avg_user_rating_arr']  # average ratings

    userID_arr = test_rating_df['userID'].values
    movieID_arr = test_rating_df['movieID'].values
    rating_arr = test_rating_df['rating'].values  # True ratings
    pred_rating_ls = []

    A_item = A.copy()
    # Replace missing values (unrated movies) with the user's average rating (helpful for similarity)
    for user_idx, col in enumerate(A_item.T):
        col[col == 0] = avg_user_rating_arr[user_idx]

    # Make prediction for each (user, movie) pair
    for userID, movieID in zip(userID_arr, movieID_arr):

        try:
            user_col = model_params['userID_hash'][userID]  # column index in A that represents userID
            movie_row = model_params['movieID_hash'][movieID]  # row index in A that represents movieID
            user_rated_ls = model_params['user_rated_dict'][user_col]  # list of movies that userID has rated
        except KeyError as err:
            print('User/Movie not seen before', err)
            pred_rating_ls.append(3)  # cheating
            continue

        # Get the k most similar movies that userID has rated
        k_nearest = _knn(movie_row, k, A_item, user_rated_ls, avg_user_rating_arr)

        if not k_nearest:
            # no neighbors
            prediction = 0
        else:
            num = denom = 0
            # Prediction is weighted
            for sim, movie_row_i in k_nearest:
                if np.isnan([sim]):
                    continue
                num += sim * A[movie_row_i, user_col]
                denom += abs(sim)
            if num == 0:
                prediction = 3  # a bit more cheating
            else:
                num = num * -1 if num < 0 else num
                prediction = num / denom
                prediction = 3 if prediction < 1 else prediction  # just a little more cheating
        if np.isnan([prediction]):
            print('NaN prediction')
            prediction = 3  # cheating
        pred_rating_ls.append(prediction)

    # Calculate MSE
    pred_rating_arr = np.array(pred_rating_ls)
    mse = mean_squared_error(rating_arr, pred_rating_arr)

    return pred_rating_arr, mse


def predict_user(test_rating_df, k, model_params):
    """Returns array of predicted value for each (user, movie) pair in test_rating_df along with MSE. Does so using the
    User-Based KNN Collaborative Filtering algorithm.

        Arguments:
            test_rating_df -- DataFrame storing labeled rating data to predict. cols - 'userID', 'movieID', 'rating'
            k -- integer representing number of neighbors to consider - number of movies that the user has already rated
                taken into consideration when predicting the rating of the user on some movie it has not rated.
            model_params -- dictionary containing existing rating data user to calculate a prediction for each user and
                movie pair.

        Returns:
            pred_rating_arr -- np array consisting of the predictions for each row of (user, movie) in test_rating_df.
            mse -- float representing the mean squared error of the calculated predictions.
    """
    # Predict userID's rating of movieID
    A = model_params['A']  # movie-user matrix
    avg_movie_rating_arr = model_params['avg_movie_rating_arr']  # average ratings

    userID_arr = test_rating_df['userID'].values
    movieID_arr = test_rating_df['movieID'].values
    rating_arr = test_rating_df['rating'].values  # True ratings
    pred_rating_ls = []

    A_user = A.T.copy()
    # Replace missing values (unrated movies) with the user's average rating (helpful for similarity)
    for movie_idx, row in enumerate(A_user.T):
        row[row == 0] = avg_movie_rating_arr[movie_idx]

    # Make prediction for each (user, movie) pair
    for userID, movieID in zip(userID_arr, movieID_arr):

        try:
            user_col = model_params['userID_hash'][userID]  # column index in A that represents userID
            movie_row = model_params['movieID_hash'][movieID]  # row index in A that represents movieID
            movie_rated_by_ls = model_params['movie_rated_by_dict'][movie_row]  # list of movies that userID has rated
        except KeyError as err:
            print('User/Movie not seen before', err)
            pred_rating_ls.append(3)  # cheating
            continue

        # Get the k most similar movies that userID has rated
        k_nearest = _knn(user_col, k, A_user, movie_rated_by_ls, avg_movie_rating_arr)

        if not k_nearest:
            # no neighbors
            prediction = 0
        else:
            num = denom = 0
            # Prediction is weighted
            for sim, user_col_i in k_nearest:
                if np.isnan([sim]):
                    continue
                num += sim * A[movie_row, user_col_i]
                denom += abs(sim)
            if num == 0:
                prediction = 3  # a bit more cheating
            else:
                num = num * -1 if num < 0 else num
                prediction = num / denom
                prediction = 3 if prediction < 1 else prediction  # just a little more cheating
        if np.isnan([prediction]):
            print('NaN prediction')
            prediction = 3  # cheating
        pred_rating_ls.append(prediction)

    # Calculate MSE
    pred_rating_arr = np.array(pred_rating_ls)
    mse = mean_squared_error(rating_arr, pred_rating_arr)

    return pred_rating_arr, mse


def _knn(row, k, A, rated_ls, avg_rating_arr):
    """Return the k movies/users most similar to movie_row/user_row.

        Arguments:
            row -- integer representing hashed row value of the movieID/userID of interest
            k -- number of closest movies/users to return
            A -- 2D numpy array representing the movie-user matrix
            rated_ls -- list of movie_row/user_row indexes that the userID/movie of interest has rated/been rated by
            avg_rating_arr -- np array of average user/movie ratings (length = num of users/movies)

        Returns:
            sim_ls -- list of (similarity score, movie_row) pairs. The k most similar movies/users

    This algorithm computes the adjusted cosine similarity between movie_row/user_row and every movie/user in
    rated_ls without looping through every movie/user. Transforming the repeated vector operations into matrix
    multiplications allows use of numpy's parallelization.
    This function can execute item-based and user-based similarity. For the case of user-based similarity the
    movie-user matrix is transposed such that rows represent users and columns represent movies.
    """
    num_ratings = len(rated_ls)
    # print(f'Num Ratings: {num_ratings}')
    if num_ratings == 0:
        # User hasn't rated any other movies
        return None

    # Only want to consider movies/users that the user/movie has rated/been rated by
    # Extract them from the matrix
    # Also, subtract the user/movie's average from each row. Note, this makes all missing ratings equal zero because we
    # initially set these values equal to the average
    idx_ls = [row] + rated_ls
    M = A[idx_ls] - avg_rating_arr

    star_0 = M[0]  # Ratings of the movie/user of interest

    star = star_0.reshape((len(star_0), 1))
    B = (star * M.T) / M.T
    mask_arr = np.ma.masked_invalid(B)
    mask = mask_arr.mask
    B[mask] = 0
    v_star = np.dot(star_0, B)  # 'variance' of star with respect to i (v_1)

    C = (M.T * star) / star
    mask_arr = np.ma.masked_invalid(C)
    mask = mask_arr.mask
    C[mask] = 0
    V_i = np.matmul(M, C)  # along diagonals is the 'variance' of i with respect to star (v_2)

    cov_vec = np.dot(M, M[0])  # 'cov' of star and i

    sim_ls = []
    for i in range(len(M)):
        if i == 0:
            continue
        row_i = rated_ls[i - 1]
        var_1 = v_star[i]
        var_2 = V_i[i, i]
        cov = cov_vec[i]
        if var_1 == 0 or var_2 == 0:
            sim = 0
        else:
            sim = cov / np.sqrt(var_1 * var_2)
        sim_ls.append((sim, row_i))
    sim_ls.sort(reverse=True)

    return sim_ls[:k]
