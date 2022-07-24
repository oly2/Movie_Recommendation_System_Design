import pandas as pd
import numpy as np
from zipfile import ZipFile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import json, os

curdir = os.getcwd()
f = open(os.path.join(curdir, 'moviesystem/data/reviews.json'))
data = json.load(f)
f.close()
df_review = pd.DataFrame(data)
df_review.source.unique()
# deliminate rating = -1
# merge IMDB rating and tomato rating (no tomato rating data currently, but plan to finish finally)
df_review = df_review[df_review['rating'] != -1]
df_review = df_review[df_review['source'] != "Rotten Tomatoes"]

#reset index for movies and users
user_ids = df_review['userID'].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}
movie_ids = df_review['movieID'].unique().tolist()
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
df_review["user"] = df_review['userID'].map(user2user_encoded)
df_review["movie"] = df_review['movieID'].map(movie2movie_encoded)

num_users = len(user2user_encoded)
num_movies = len(movie_encoded2movie)
df_review["rating"] = df_review["rating"].values.astype(np.float32)
# min and max ratings will be used to normalize the ratings later
min_rating = min(df_review["rating"])
max_rating = max(df_review["rating"])

df_review = df_review.sample(frac=1, random_state=42)
x = df_review[["user", "movie"]].values
# Normalize the targets between 0 and 1. Makes it easy to train.
y = df_review["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
# Assuming training on 90% of the data and validating on 10%.
train_indices = int(0.9 * df_review.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:],
)

df_review = df_review.sample(frac=1, random_state=42)
x = df_review[["user", "movie"]].values
# Normalize the targets between 0 and 1. Makes it easy to train.
y = df_review["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
# Assuming training on 90% of the data and validating on 10%.
train_indices = int(0.9 * df_review.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:],
)

EMBEDDING_SIZE = 50


class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_movies, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.movie_embedding = layers.Embedding(
            num_movies,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.movie_bias = layers.Embedding(num_movies, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)
        # Add all the components (including bias)
        x = dot_user_movie + user_bias + movie_bias
        # The sigmoid activation forces the rating to between 0 and 1
        return tf.nn.sigmoid(x)


model = RecommenderNet(num_users, num_movies, EMBEDDING_SIZE)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(lr=0.001)
)

history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=64,
    epochs=30,
    verbose=1,
    validation_data=(x_val, y_val),
)

f = open(os.path.join(curdir, 'moviesystem/data/movies_rating_updated.json'))
data = json.load(f)
f.close()
movie_df = pd.DataFrame(data)

# Let us get a user and see the top recommendations.
def recommend(user_id):
    movies_watched_by_user = df_review[df_review.userID == user_id]
    movies_not_watched = movie_df[
        ~movie_df["movieID"].isin(movies_watched_by_user.movieID.values)
    ]["movieID"]
    movies_not_watched = list(
        set(movies_not_watched).intersection(set(movie2movie_encoded.keys()))
    )
    movies_not_watched = [[movie2movie_encoded.get(x)] for x in movies_not_watched]
    user_encoder = user2user_encoded.get(user_id)
    user_movie_array = np.hstack(
        ([[user_encoder]] * len(movies_not_watched), movies_not_watched)
    )
    ratings = model.predict(user_movie_array).flatten()
    top_ratings_indices = ratings.argsort()[-10:][::-1]
    recommended_movie_ids = [
        movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices
    ]

    recommended_movies = movie_df[movie_df["movieID"].isin(recommended_movie_ids)]

    final_rec_result = recommended_movies[['movieID', 'name', 'genres', 'ratingFromIMDB']]
    final_rec_result['userID'] = user_id

    return final_rec_result

def recommend_all():
    # dropping ALL duplicate values
    df_proc = df_review.drop_duplicates(subset =["userID"], keep = 'first')
    each_result_list = []
    for i, x in enumerate(df_proc.userID):
        each_result = recommend(x)
        print(i)
        each_result_list.append(each_result)
    all_result = pd.concat(each_result_list, axis=0)
    #all_result.to_csv('D:/E/Courses/CS 411/PT1/sp22-cs411-team014-JJYY-main-final/moviesystem/data/all_recom_has_reviewed.csv', index=False)
    return all_result

#recommend_all()
def recom_for_notreview(df_review, df_user, df_movie):
    df_user = df_user[~df_user["userID"].isin(df_review.userID)]
    final_result = []
    for user in range(len(df_user)):
        for i in range(3):
            # find user's prefer genre
            pregenre = df_user.iloc[user,3].split(",")[i]
            pregenre = pregenre.replace(" ", "")
            recmovie_list_all = df_movie.loc[df_movie['genres'].str.contains(pregenre)]
            if recmovie_list_all.empty:
                continue
            # find 2 movies from all movies that belong to that genre
            recmovie_list = recmovie_list_all.sample(2)
            recmovie_list['userID'] = df_user.iloc[user]['userID']
            recmovie_list = recmovie_list.values.tolist()
            final_result += recmovie_list
            df_movie = df_movie[pd.to_numeric(df_movie['releaseDate'], errors='coerce').notnull()]
        extramovie_list = df_movie[(df_movie['releaseDate'].astype(str).astype(float)>2000)
                                    & (df_movie['ratingFromIMDB']>8)].sample()
        extramovie_list['userID'] = df_user.iloc[user]['userID']
        extramovie_list = extramovie_list.values.tolist()
        final_result += extramovie_list
    final_result_df = pd.DataFrame(final_result, columns = ['movieID', 'name', 'releaseDate',   'duration',
                                                            'genres',   'ratingFromTomato', 'contentRating',
                                                            'ratingFromIMDB',   'directorID', 'userID'])
    final_result_df.to_csv(os.path.join(curdir, 'moviesystem/data/all_recom_no_review.csv'), index=False)
    return final_result_df


def retrieveResForUser(userID):
    """
    (function) Retrieves top recommended movies for the given userID. Returns a list of movieID
    """

    df_user = pd.read_csv(os.path.join(curdir, 'moviesystem/data/User.csv'))
    f = open(os.path.join(curdir, 'moviesystem/data/reviews.json'))
    data = json.load(f)
    f.close()
    df_review = pd.DataFrame(data)
    df_user_no_comment = df_user[~df_user["userID"].isin(df_review.userID)]
    recomsForNoReviews = pd.read_csv(os.path.join(curdir, 'moviesystem/data/all_recom_no_review.csv'))
    ls = []
    for i, r in recomsForNoReviews.iterrows():
        if (r["userID"] == userID):
            ls.append(r["movieID"])
    if (len(ls) == 0):
        recomsForReviews = pd.read_csv(os.path.join(curdir, 'moviesystem/data/all_recom_has_reviewed.csv'))
        c = 0
        for i, r in recomsForReviews.iterrows():
            if (r["userID"] == userID):
                ls.append(r["movieID"])
    print(ls)
    return ls
