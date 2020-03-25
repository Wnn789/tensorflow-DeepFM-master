import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Embedding, Reshape, Activation, Input, Dense, Flatten, Dropout
from keras.layers.merge import Dot, multiply, concatenate
from keras.utils import np_utils
from keras.utils.data_utils import get_file
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import skipgrams
from collections import defaultdict

def get_mapping(series):  #获取映射,没有重复项
    occurances = defaultdict(int)   #defaultdict函数表示当字典中的key不存在但被查找时，返回一个0（int来决定）来代替报错
    for element in series:
        occurances[element] += 1
    mapping = {}
    i = 0
    for element in occurances:
        i += 1
        mapping[element] = i

    return mapping




def get_data():
    data = pd.read_csv("data/ratings.csv")

    mapping_work = get_mapping(data["movieId"])  #将“movieId”存入字典中，将顺序序号作为键，movieId为值

    data["movieId"] = data["movieId"].map(mapping_work)

    mapping_users = get_mapping(data["movieId"])

    data["movieId"] = data["movieId"].map(mapping_users)

    percentil_80 = np.percentile(data["timestamp"], 80)  #时间步从小到大排列，取其百分之80位置的数

    print(percentil_80)

    print(np.mean(data["timestamp"]<percentil_80))

    print(np.mean(data["timestamp"]>percentil_80))

    cols = ["userId", "movieId", "rating"]

    train = data[data.timestamp<percentil_80][cols]

    print(train.shape)

    test = data[data.timestamp>=percentil_80][cols]

    print(test.shape)

    max_user = max(data["userId"].tolist() )
    max_work = max(data["movieId"].tolist() )


    return train, test, max_user, max_work, mapping_work


# keras.layers.embeddings.Embedding(input_dim, output_dim, embeddings_initializer='uniform', embeddings_regularizer=None, activity_regularizer=None, embeddings_constraint=None, mask_zero=False, input_length=None)
# 参数
#
# input_dim：大或等于0的整数，字典长度，即输入数据最大下标+1
# output_dim：大于0的整数，代表全连接嵌入的维度
# embeddings_initializer: 嵌入矩阵的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers
# embeddings_regularizer: 嵌入矩阵的正则项，为Regularizer对象
# embeddings_constraint: 嵌入矩阵的约束项，为Constraints对象
# mask_zero：布尔值，确定是否将输入中的‘0’看作是应该被忽略的‘填充’（padding）值，该参数在使用递归层处理变长输入时有用。设置为True的话，模型中后续的层必须都支持masking，否则会抛出异常。如果该值为True，则下标0在字典中不可用，input_dim应设置为|vocabulary| + 2。
# input_length：当输入序列的长度固定时，该值为其长度。如果要在该层后接Flatten层，然后接Dense层，则必须指定该参数，否则Dense层的输出维度无法自动推断。
# 输入shape
# 形如（samples，sequence_length）的2D张量
# 输出shape
# 形如(samples, sequence_length, output_dim)的3D张量
#
# 较为费劲的就是第一句话：
# 嵌入层将正整数（下标）转换为具有固定大小的向量，如[[4],[20]]->[[0.25,0.1],[0.6,-0.2]]


def get_model_1(max_work, max_user):
    dim_embedddings = 30
    bias = 3
    # inputs
    w_inputs = Input(shape=(1,), dtype='int32')
    w = Embedding(max_work+1, dim_embedddings, name="work")(w_inputs)

    # context
    u_inputs = Input(shape=(1,), dtype='int32')
    u = Embedding(max_user+1, dim_embedddings, name="user")(u_inputs)
    o = multiply([w, u])
    o = Dropout(0.5)(o)
    o = Flatten()(o)
    o = Dense(1)(o)

    rec_model = Model(inputs=[w_inputs, u_inputs], outputs=o)
    rec_model.summary()
    rec_model.compile(loss='mae', optimizer='adam', metrics=["mae"])

    return rec_model


def get_model_2(max_work, max_user):
    dim_embedddings = 30
    bias = 1
    # inputs
    w_inputs = Input(shape=(1,), dtype='int32')
    w = Embedding(max_work+1, dim_embedddings, name="work")(w_inputs)
    w_bis = Embedding(max_work + 1, bias, name="workbias")(w_inputs)

    # context
    u_inputs = Input(shape=(1,), dtype='int32')
    u = Embedding(max_user+1, dim_embedddings, name="user")(u_inputs)
    u_bis = Embedding(max_user + 1, bias, name="userbias")(u_inputs)
    o = multiply([w, u])
    o = concatenate([o, u_bis, w_bis])
    o = Dropout(0.5)(o)
    o = Flatten()(o)
    o = Dense(1)(o)

    rec_model = Model(inputs=[w_inputs, u_inputs], outputs=o)
    rec_model.summary()
    rec_model.compile(loss='mae', optimizer='adam', metrics=["mae"])

    return rec_model

def get_model_3(max_work, max_user):
    dim_embedddings = 30
    bias = 1
    # inputs
    w_inputs = Input(shape=(1,), dtype='int32')
    w = Embedding(max_work+1, dim_embedddings, name="work")(w_inputs)
    w_bis = Embedding(max_work + 1, bias, name="workbias")(w_inputs)

    # context
    u_inputs = Input(shape=(1,), dtype='int32')
    u = Embedding(max_user+1, dim_embedddings, name="user")(u_inputs)
    u_bis = Embedding(max_user + 1, bias, name="userbias")(u_inputs)
    o = multiply([w, u])
    o = Dropout(0.5)(o)
    o = concatenate([o, u_bis, w_bis])
    o = Flatten()(o)
    o = Dense(10, activation="relu")(o)
    o = Dense(1)(o)

    rec_model = Model(inputs=[w_inputs, u_inputs], outputs=o)
    rec_model.summary()
    rec_model.compile(loss='mae', optimizer='adam', metrics=["mae"])

    return rec_model

def get_array(series):
    return np.array([[element] for element in series])

#计算两个向量的余弦相似度
def cosine_similarity(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return 0
    else:
        return dot_product / ((normA**0.5)*(normB**0.5))


#计算与所选电影类别最相近的电影列表
def get_movie_genres(movie_title):
    data = pd.read_csv("data/movies.csv")
    i = 0
    most_same_genres = {}
    same_genres = {}
    mapping_movie = get_mapping_seq(data["genres"])     #将电影体裁描述放入一个字典中，顺序号为键，描述为值
    for key in mapping_movie:                       #此循环将电影的描述切分开放入列表中
        mapping_movie[key] = mapping_movie[key].split('|')
    for key in mapping_movie:                       #此循环将所选电影的体裁描述和所有电影的体裁描述对比，并获取相同的部分
        same_genres[key] = [genres for genres in mapping_movie[movie_title] if genres in mapping_movie[key]]
    for genres in same_genres:                      #此循环将相同的体裁描述数目大于3的电影类别选出
        if len(same_genres[genres])>=3:
            most_same_genres[genres] = same_genres[genres]
    msg_len = len(most_same_genres)
    checked_movie = [0 for x in range(0,msg_len)]   #创建一个列表，将与所选电影体裁最接近的电影编号以列表形式输出
    for key in most_same_genres:
        checked_movie[i] = key
        i+=1
    return checked_movie

def get_mapping_seq(series):  #获取电影正向序列
    occurances = defaultdict(int)   #defaultdict函数表示当字典中的key不存在但被查找时，返回一个0（int来决定）来代替报错
    for element in series:
        occurances[element] += 1
    mapping = {}
    i = 0
    for element in occurances:
        i += 1
        mapping[i] = element

    return mapping

