import pandas as pd
import numpy as np
from db_connection.connection import load_to_db

movies = pd.read_csv('D:/osspTeamProject/data/ml-25m.csv', converters={"genres": lambda x: x.replace("'", "").strip("[]").split(", ")})

# genre -> one-hot encoding, as a dataframe
genre_list = movies.genres
genre_set = sorted(list(set([genre[i] for genre in genre_list for i in range(len(genre))])))
genre_encoding = []  # one-hot encoding

for genres in genre_list:
    tmp = []
    for genre in genre_set:
        if genre in genres:
            tmp.append(1)
        else:
            tmp.append(0)
    genre_encoding.append(tmp)

genre_encoding = np.array(genre_encoding)  # 배열로 변환

genres = pd.DataFrame(data=genre_encoding, columns=genre_set)
genres.set_index(movies.movieId, inplace=True)

ratings = pd.read_csv('D:/osspTeamProject/data/ratings.csv')
ratings.rating = ratings.rating.apply(lambda x: 1 if x >= 4.5 else 0)
ratings.rename(columns={'rating': 'is_click'}, inplace=True)

movies.drop('genres', axis=1, inplace=True)
total = movies.merge(ratings, on='movieId', how="inner")
total = total.merge(genres, on='movieId', how='inner')

# 열 재배치
col1 = ['movieId', 'userId', 'title']
col2 = total.columns[3:].to_list()
new_col = col1 + col2
total = total[new_col]

total.sort_values(by=['userId', 'timestamp'], ascending=True, inplace=True)
total.reset_index(drop=True, inplace=True)

load_to_db(total, 'data')