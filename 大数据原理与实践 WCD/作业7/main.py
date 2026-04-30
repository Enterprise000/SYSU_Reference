import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

movie = pd.read_csv('movies.dat',sep = '::',header=None,names=['movie_id','title','genres'],engine='python',encoding= 'latin1')
rating = pd.read_csv('ratings.dat',sep = '::',header=None,names=['user_id','movie_id','rating','timestamp'],engine='python',encoding= 'latin1')
user = pd.read_csv('users.dat',sep = '::',header=None,names=['user_id','gender','age','occupation','zip'],engine='python',encoding= 'latin1')
print(movie.head())
print(rating.head())
print(user.head())

# 协同过滤
user_rating = rating.pivot_table(index='user_id',columns='movie_id',values='rating')
# print(user_rating.head())
user_similarity = cosine_similarity(user_rating.fillna(0))
similarity_df = pd.DataFrame(user_similarity,index=user_rating.index, columns=user_rating.index)
# print(similarity_df.head)
user_id = 1  # 可以修改
# 找到最相近的五个用户
users = similarity_df[user_id].sort_values(ascending=False).index[1:6]
rated = user_rating.loc[user_id].index
similar_ratings = user_rating.loc[users,rated]
recommend = similar_ratings.mean(axis=0).sort_values(ascending=False).head(20).index.tolist()

user_ratings = rating[rating['user_id'] == 1]
movie_like = user_ratings[user_ratings['rating']>4]['movie_id']

TP = len(set(movie_like).intersection(set(recommend)))
recall = TP / len(movie_like)
# 打印推荐列表和用户喜欢列表
print("推荐电影 ID:", recommend)
print("用户喜欢电影 ID:", movie_like)

# 检查交集
intersection = set(movie_like).intersection(set(recommend))
print("交集:", intersection)
print(f"召回率：{recall:.2f}")

