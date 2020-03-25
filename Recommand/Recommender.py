import numpy as np
import pandas as pd
import pickle
import heapq

# # # 读取基因组数据,并保存矩阵
# #tag = 1128
# #movie = 62423,206499
# #genome_orig = (15584448, 3)
# genome = pd.read_csv('E:\ML\代码自习\Movie_Recommendation\movie_data\ml-25m\genome-scores.csv    ', sep=',', header=0,
#                      engine='python')
# genome_orig = genome.values
# genome_mat = np.zeros((62424,1130))
#
# movieid_ = 1
# genome_row = 0
# for row in genome_orig:
#     if row[0] == movieid_:
#         genome_mat[genome_row][0]=row[0]
#         genome_mat[genome_row][int(row[1])]=row[2]
#     else:
#         genome_row+=1
#         movieid_ = row[0]
#         genome_mat[genome_row][0] = row[0]
#         genome_mat[genome_row][int(row[1])] = row[2]
#
#
# pickle.dump((genome_mat), open('genome_mat.p', 'wb'))

genome_mat = pickle.load(open('genome_mat.p', mode='rb'))

movie = pd.read_csv('E:\ML\代码自习\Movie_Recommendation\movie_data\ml-25m\movies.csv', sep=',', header=0,engine='python')
movie_orig = movie.values #(62423, 3)
movieName_id = {}
movieId_name = {}
for row in movie_orig:
    movieName_id[row[1]] = row[0]
    movieId_name[row[0]] = row[1]



def recommand_movie(watched_movie):
    count = 1
    movie_id = []
    recommand_movie_list_id = []
    for movieName in movieName_id:
        if watched_movie in movieName:
            movie_id.append(movieName_id[movieName])
    for row in genome_mat:
        if row[0] in movie_id:
            print('我看过的电影有：', movieId_name[row[0]])
            row = row.tolist()
            max_rating_watched = list(map(row.index, heapq.nlargest(10, row)))
            break
    for row in genome_mat:
        row = row.tolist()
        max_rating_movies = list(map(row.index, heapq.nlargest(10, row)))
        if len(list(x for x in max_rating_movies if x in max_rating_watched))>5:
            recommand_movie_list_id.append(row[0])
    print('系统推荐的电影有:')
    for x in recommand_movie_list_id:
        if x !=0:
            print(count,'.',movieId_name[x])
            count+=1
            if count>10:
                break


def movie_choose(watched_movie):
    movie_id = []
    movie_list_id = []
    for movieName in movieName_id:
        if watched_movie in movieName:
            movie_id.append(movieName_id[movieName])
    for row in genome_mat:
        if row[0] in movie_id:
            movie_list_id.append(row[0])
    print("数据中可选用的电影id有")
    for i in movie_list_id:
        print(movieId_name[i])








watched_movie = 'Avengers: Infinity War - Part I (2018)'


#选出可用的电影名称
# movie_choose(watched_movie)

#进行电影推荐
recommand_movie(watched_movie)








