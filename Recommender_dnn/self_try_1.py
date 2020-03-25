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
import plotly
import plotly.graph_objs as go

def get_mapping(series):  #将series作为键，值为序列
    occurances = defaultdict(int)   #defaultdict函数表示当字典中的key不存在但被查找时，返回一个0（int来决定）来代替报错
    a = 0
    for element in series:
        a += 1
        occurances[element] += 1    #occurances将输入的列为键，1为值                                         defaultdict(<class 'int'>, {1: 1, 3: 1, 6: 1, 47: 1, 50: 1, 70: 1})
    mapping = {}
    i = 0
    for element in occurances:
        i += 1
        mapping[element] = i       #mapping字典，将输入作为键，值为序列

    return mapping

data = pd.read_csv("data/ratings.csv")


mapping_work = get_mapping(data["movieId"])
# print(mapping_work)   #{1: 1, 3: 2, 6: 3, 47: 4, 50: 5, 70: 6}
data["movieId"] = data["movieId"].map(mapping_work)  #（pandas中的map）将列表data["movieId"]中的值，用字典mapping_work中的值来代替    #(python中的map)如果函数是 None，自动假定一个‘identity’函数,这时候就是模仿 zip()函数，这时候 None 类型不是一个可以调用的对象。所以他没法返回值。目的是将多个列表相同位置的元素归并到一个元组。
print('data["movieId"]',data["movieId"][1000])
# print(type(data["movieId"]))
mapping_users = get_mapping(data["movieId"])

data["movieId"] = data["movieId"].map(mapping_users)
print('data["movieId"]',data["movieId"])
print('work,users',mapping_work,'\n\t',mapping_users)
percentil_80 = np.percentile(data["timestamp"], 80)

# print(percentil_80)

print(np.mean(data["timestamp"] < percentil_80))

print(np.mean(data["timestamp"] > percentil_80))

cols = ["userId", "movieId", "rating"]

train = data[data.timestamp < percentil_80][cols]
print('data.timestamp',data.timestamp)  #.后面加列名，可以获取相应的列

print(train)

test = data[data.timestamp >= percentil_80][cols]

print(test.shape)

max_user = max(data["userId"].tolist())
max_work = max(data["movieId"].tolist())
print(max_user)
print(max_work)

def get_array(series):
    return np.array([[element] for element in series])

print([get_array(train["movieId"]), get_array(train["userId"])])