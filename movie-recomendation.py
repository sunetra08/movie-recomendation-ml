import numpy as np
import difflib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

df = pd.read_csv('netflix_titles.csv')
# print(df.info())

selective = ['cast', 'director', 'genres', 'keywords', 'tagline']
for feature in selective:
    df[feature] = df[feature].fillna('')

combine = df['cast']+' '+df['director']+' ' + \
    df['genres']+' ' + df['keywords']+' ' + df['tagline']

# print(combine)
vectorizer = TfidfVectorizer()
feature_vector = vectorizer.fit_transform(combine)
# print(feature_vector)

similarity = cosine_similarity(feature_vector)
# print(similarity.shape)

movie_name = input('enter movie name:  ')
list_of_all_title = df['title'].tolist()
find_closer = difflib.get_close_matches(movie_name, list_of_all_title)
# print(find_closer)
close_match = find_closer[0]
index_of = df[df.title == close_match]['index'].values[0]
# print(index_of)

similarity_score = list(enumerate(similarity[index_of]))
# print(similarity_score)
similar = sorted(similarity_score, key=lambda x: x[1], reverse=True)
# print(similar)
print('movie suggestion for user:  ')
i = 1
for movie in similar:
    index = movie[0]
    title_from_index = df.loc[index, 'title']
    if i < 30:
        print(i, ' ', title_from_index)
        i += 1
