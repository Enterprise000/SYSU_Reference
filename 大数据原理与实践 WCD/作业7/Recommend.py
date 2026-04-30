import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity


movie = pd.read_csv('movies.dat',sep = '::',header=None,names=['movie_id','title','genres'],engine='python',encoding= 'latin1')
rating = pd.read_csv('ratings.dat',sep = '::',header=None,names=['user_id','movie_id','rating','timestamp'],engine='python',encoding= 'latin1')
user = pd.read_csv('users.dat',sep = '::',header=None,names=['user_id','gender','age','occupation','zip'],engine='python',encoding= 'latin1')
print(rating.head())

binarizer = MultiLabelBinarizer()
movie_genre = binarizer.fit_transform(movie['genres'].str.split('|'))
movie_genre_df = pd.DataFrame(movie_genre,columns=binarizer.classes_,index=movie['movie_id'])
# print(movie_genre_df.head())

movie_similarity = cosine_similarity(movie_genre_df)
movie_similarity_df = pd.DataFrame(movie_similarity, columns=movie['movie_id'],index = movie['movie_id'])
# print(movie_similarity_df.head())

user_ratings = rating[rating['user_id'] == 1]
movie_like = user_ratings[user_ratings['rating']>4]['movie_id']
# print(movie_like)
movie_id = movie_like.values.tolist()[13]
print("movieid:",movie_id)
similar_movie = movie_similarity_df[movie_id].sort_values(ascending=False)
recommend_movie = similar_movie.head(30).index.tolist()
print("推荐电影 ID:", recommend_movie)
print("用户喜欢电影 ID:", movie_like)
print("交集：",set(movie_like).intersection(set(recommend_movie)))
TP = len(set(movie_like).intersection(set(recommend_movie)))
recall = TP / len(movie_like)

print(f"召回率：{recall:.2f}")

