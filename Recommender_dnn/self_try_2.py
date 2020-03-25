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


def dict_change(dict): #将字典的键对值互换
    d2 = {}
    for key, value in dict.items():
        d2[value] = key
    return d2


def get_mapping(series):  #获取映射
    occurances = defaultdict(int)   #defaultdict函数表示当字典中的key不存在但被查找时，返回一个0（int来决定）来代替报错
    for element in series:
        occurances[element] += 1
    mapping = {}
    i = 0
    for element in occurances:
        i += 1
        mapping[i] = element

    return mapping

def get_movie_genres(movie_title):
    data = pd.read_csv("data/movies.csv")
    i = 0
    most_same_genres = {}
    same_genres = {}
    mapping_movie = get_mapping(data["genres"])     #将电影体裁描述放入一个字典中，顺序号为键，描述为值
    for key in mapping_movie:                       #此循环将电影的描述切分开放入列表中
        mapping_movie[key] = mapping_movie[key].split('|')
    for key in mapping_movie:                       #此循环将所选电影的体裁描述和所有电影的体裁描述对比，并获取相同的部分
        same_genres[key] = [genres for genres in mapping_movie[movie_title] if genres in mapping_movie[key]]
    for genres in same_genres:                      #此循环将相同的体裁描述数目大于3的电影类别选出
        if len(same_genres[genres])>=3:
            most_same_genres[genres] = same_genres[genres]
    msg_len = len(most_same_genres)
    checked_movie = [0 for x in range(0,msg_len)]   #创建一个列表，将与所选电影体裁最接近的电影编号以列表形式输出
    print(len(most_same_genres),len(most_same_genres),)
    for key in most_same_genres:
        checked_movie[i] = key
        i+=1
    print(checked_movie)
    return checked_movie


get_movie_genres(9)


