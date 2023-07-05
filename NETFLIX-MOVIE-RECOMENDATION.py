import numpy as np
import difflib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


netflix = pd.read_csv('netflix_movie.csv')

netflix['index'] = netflix.reset_index().index + 1

# Rearrange the columns to have 'index_col' as the first column
cols = netflix.columns.tolist()
cols = ['index'] + cols[:-1]
netflix = netflix[cols]


# print(netflix.head())
selective = ['listed_in', 'description', 'cast', 'director', 'type']
for feature in selective:
    netflix[feature] = netflix[feature].fillna('')

combile = netflix['listed_in']+' ' + netflix['description']+' ' + \
    netflix['cast']+' ' + netflix['director']+' ' + netflix['type']

vectorizer = TfidfVectorizer()
feature_vector = vectorizer.fit_transform(combile)
# print(feature_vector)


similarity = cosine_similarity(feature_vector)

movie = input('enter movie or series name: ')
list_of_movies = netflix['title'].tolist()
find_close = difflib.get_close_matches(movie, list_of_movies, cutoff=0.5)
for closer in find_close:
    index_of = netflix[netflix.title == closer]['index'].values
    if len(index_of) > 0:
        index_of = index_of[0]
        similarity_score = list(enumerate(similarity[index_of]))
        similar = sorted(similarity_score, key=lambda x: x[1], reverse=True)
        print('Similar movies to', closer, ':')
        i = 0
        for m in similar:
            index = m[0]
            title = netflix.loc[index, 'title']
            description = netflix.loc[index, 'description']
            if i < 20:
                print(i, ' ', title, ' - ', description)
                i += 1
        break
