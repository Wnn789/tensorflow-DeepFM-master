from utils import *
import pickle
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
#import plotly.plotly as py
import chart_studio.plotly as py
import plotly
import plotly.graph_objs as go
from imp import reload
import sys
import heapq
reload(sys)
# sys.setdefaultencoding('utf-8')



train, test, max_user, max_work, _ = get_data()

movies = pd.read_csv("data/movies.csv")

movie_title = dict(zip(movies["movieId"], movies["title"]))  #将电影名称提出放入字典

model = get_model_3(max_user=max_user, max_work=max_work)
model.load_weights("model_3.h5")

embedding_work = model.get_layer("work").get_weights()[0]

print(embedding_work,embedding_work.shape)

mapping_work = pickle.load(open("mapping_work.pkl", "rb")) #反序列化对象，将文件中的数据解析为一个python对象。file中有read()接口和readline()接口

reverse_mapping = dict((v,k) for k,v in mapping_work.items())   #反向映射

embedding = {}
i = 0
for id in movie_title:
    if id in mapping_work:
        embedding[id] = embedding_work[mapping_work[id], :]


list_titles = []
list_embeddings = []

for id in embedding:
    list_titles.append(movie_title[id])
    list_embeddings.append(embedding[id])

print('list_titles',list_titles)

matrix_embedding = np.array(list_embeddings)
# ##################################################################################################


# op2=np.linalg.norm(vector1-vector2) linalg = linear(线性) + algebra(代数) norm 表示范数
#以下是实验的计算最近距离的方法
movie_watched = 9                           #选择一个电影
[rows, cols] = matrix_embedding.shape       #获取嵌入矩阵的行和列
list_RESULT =[0 for x in range(0,rows)]     #生成一个和嵌入矩阵行相同的列表
print([rows, cols])
for i in range(rows):
    # list_RESULT[i] = np.linalg.norm(matrix_embedding[i]-matrix_embedding[movie_watched])          #将嵌入矩阵的每一行（电影）数据和所选行（电影）做差并求范数，计算其欧氏距离
    list_RESULT[i] = cosine_similarity(matrix_embedding[i], matrix_embedding[movie_watched])        #计算其余弦距离
min_num_index=map(list_RESULT.index, heapq.nlargest(5,list_RESULT))  #求最小(nsmalist,大：nlargest)值的索引，找到和所选电影最近的三个电影
list_movie_1 =list(min_num_index)


#此时注意map的用法！！！！
# 这是由于，map函数返回的，是一个“可迭代对象”。

# 这种对象，被访问的同时，也在修改自己的值。 类似于 a = a+1 ,这样。对于map来说，就是每次访问，都把自己变为List中的下一个元素。
#
# 循环取得对象中的值 ，实际上是会调用内部函数__next__，将值改变，或者指向下一个元素。
#
# 当多次调用，代码认为到达终点了，返回结束，或者__next__指向空，此时可迭代对象（链表） 就算到终点了，不能再用了。
#
# 类似于 list(A_object) 或者 for num in A_object 这样的语句，就是调用了迭代器，执行了__next__,消耗了迭代对象。所以，再次使用A_object后，会发现它已经空了。

for i in list_movie_1:               #将得到的电影编号最终变为电影名称输出
    movie_title = list_titles[i]
    print('推荐的电影',movie_title)





#将权重降维和可视化
# ###################################################################################################
X_embedded = TSNE(n_components=2).fit_transform(matrix_embedding)  #降维可视化方法，——总矩阵
# np.savetxt("X_embedded.txt", X_embedded) # 缺省按照'%.18e'格式保存数据，以空格分隔
# X_embedded = np.loadtxt("X_embedded.txt")
checked_movie = get_movie_genres(9)
i = 0
for x in range(0,len(checked_movie)):
    if i==0:
        select_rows = X_embedded[checked_movie[x] - 1, :]
        i+=1
    else:
        if checked_movie[x] >9724:
            break
        select_rows = np.vstack((select_rows,(X_embedded[checked_movie[x] - 1, :])))


vis_x = X_embedded[:, 0]
vis_y = X_embedded[:, 1]
# vis_x1 = select_rows[:, 0]
# vis_y1 = select_rows[:, 1]

data = [
    go.Scatter(
        x=vis_x,
        y=vis_y,
        mode='markers',
        text=list_titles,
        marker=dict(
            color = 'blue'
        )
    )
]

# data1 = [
#     go.Scatter(
#         x=vis_x1,
#         y=vis_y1,
#         mode='markers',
#         text=list_titles,
#         marker=dict(
#             color = 'red'
#         )
#     )

# ]
data_all = data     #[data,data1]
layout = go.Layout(
    title='Movies'
)

fig = go.Figure(data=data_all, layout=layout)

plotly.offline.plot(fig, filename='movies.html')
# ####################################################################################################################################



# ##################################################################################################
# # op2=np.linalg.norm(vector1-vector2) linalg = linear(线性) + algebra(代数) norm 表示范数
# #以下是实验的计算最近距离的方法
#
# [rows, cols] = X_embedded.shape
# list_RESULT =[0 for x in range(0,rows)]
# print([rows, cols])
# for i in range(rows):
#     list_RESULT[i] = np.linalg.norm(X_embedded[i]-X_embedded[1])
# print('list_RESULT',list_RESULT)
# min_num_index=map(list_RESULT.index, heapq.nsmallest(2,list_RESULT))  #求最小值
# print('min_num_index',list(min_num_index))
# for i in min_num_index:
#     movie_title = list_titles[i]
#     print('movie_title',movie_title)
# ###################################################################################################

