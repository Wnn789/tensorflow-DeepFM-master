# import numpy as np
# import tensorflow as tf
# import scipy.io as sio
# import os
#
# with tf.Session() as sess:
#     # load the meta graph and weights
#
#     saver = tf.train.import_meta_graph('./checkpoint/model.ckpt.meta')  # 键入模型名称.meta文件
#     saver.restore(sess, tf.train.latest_checkpoint('./checkpoint/'))  # 从checkpoints中恢复最新模型
#     graph = tf.get_default_graph()  # 得到模型的默认图
#     # 按名字获得某一个参数并保存
#     #     conv1_w = sess.run(graph.get_tensor_by_name('Conv1/W_conv1/W_conv1:0'))
#     #     sio.savemat("./net/cnn_for_mnist/weights/conv1_w.mat", {"array": conv1_w})
#
#     # 从中挑选模型参数，我所需的参数是所有的trainable_variables，以及非训练参数中的moving_mean，moving_variance，根据名称索引
#     var_list = tf.trainable_variables()
#     g_list = tf.global_variables()
#     bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
#     bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
#     variable_names = var_list + bn_moving_vars
#
#     # 打印出所需参数的name，shape，info，value
#     values = sess.run(variable_names)
#     cnt = 0
#     for k, v in zip(variable_names, values):
#         print("Variable: ", k)
#         print("Shape: ", v.shape)
#         path = ('./checkpoint/parameters/' + k.name).rstrip(':0')
#         if not os.path.exists(path):
#             os.makedirs(path)
#             sio.savemat(path + '/{:s}.mat'.format(k.name.rstrip(':0').replace('/', '-')), {'array': v})  # 保存需要的参数
#         else:
#             print('path existd!', path)
#         # if 'feature_embeddings' in k.name:  # 选择性打印我所需的参数
#         #     print(v)
#         cnt += 1

#================================================================================================================================================================


# from scipy.io import loadmat
# m = loadmat('E:\ML\代码自习\Movie_Recommendation\\tensorflow-DeepFM-master\example\checkpoint\parameters\\Variable_5\\Variable_5.mat')
# # print(m)
# a = m['array']
# print(a.shape)
# for i in a:
#     print(i)
#     break


#======================================================================
import tensorflow as tf

#Prepare to feed input, i.e. feed_dict and placeholders
w1 = tf.placeholder("float", name="w1")
w2 = tf.placeholder("float", name="w2")
b1= tf.Variable(2.0,name="bias")
feed_dict ={w1:4,w2:8}

#Define a test operation that we will restore
w3 = tf.add(w1,w2)
w4 = tf.multiply(w3,b1,name="op_to_restore")
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#Create a saver object which will save all the variables


#Run the operation by feeding input
print(sess.run(w4,feed_dict))
#Prints 24 which is sum of (w1+w2)*b1

#Now, save the graph
