################################################################################################################################
# 这个文件用于从数据集中提取特征并保存到磁盘
# 输入: 
# 输出:
################################################################################################################################

import random
# import tensorflow.compat.v1 as tf
import tensorflow as tf
from tensorflow.python.platform import gfile
#tf.disable_v2_behavior()
import numpy as np
import os
#from scipy import ndimage
#from scipy.spatial.distance import cosine
#import matplotlib.pyplot as plt
#from sklearn.neighbors import NearestNeighbors
import pickle
#from tensorflow.python.platform import gfile

# 定义常量
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # 约134M

def create_inception_graph():
    """从保存的 GraphDef 文件创建图，并返回图对象。

    返回:
        保存了训练好的 Inception 网络的图，以及我们将操作的各种张量。
    """
    with tf.Graph().as_default() as graph:
        model_filename = os.path.join('imagenet', 'classify_image_graph_def.pb')
        with gfile.GFile(model_filename, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
                tf.import_graph_def(graph_def, name='', return_elements=[
                    BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
                    RESIZED_INPUT_TENSOR_NAME]))
    return graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor

def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    """在图像上运行瓶颈（特征提取）操作。

    参数:
        sess: 当前的 TensorFlow 会话。
        image_data: 图像数据。
        image_data_tensor: 输入图像数据的张量。
        bottleneck_tensor: 瓶颈张量。

    返回:
        从图像中提取的瓶颈值。
    """
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values

def iter_files(rootDir):
    """迭代遍历根目录下的所有文件。

    参数:
        rootDir: 根目录。

    返回:
        所有文件的列表。
    """
    all_files = []
    for root, dirs, files in os.walk(rootDir):
        for file in files:
            file_name = os.path.join(root, file)
            all_files.append(file_name)
        for dirname in dirs:
            iter_files(dirname)
    return all_files

# 从预训练模型的倒数第二层获取输出

img_files = iter_files('database/dataset')
#sandals_files = iter_files('uploads/dogs_and_cats/Sandals')
#shoes_files = iter_files('uploads/dogs_and_cats/Shoes')
#slippers_files = iter_files('uploads/dogs_and_cats/Slippers')
#apparel_files = iter_files('uploads/dogs_and_cats/apparel')

all_files = img_files#boots_files + shoes_files + slippers_files + sandals_files + apparel_files

random.shuffle(all_files)

# 动态确定实际图片数量
num_images = min(10000, len(all_files))
neighbor_list = all_files[:num_images]

# 保存邻居列表
with open('neighbor_list_recom.pickle','wb') as f:
        pickle.dump(neighbor_list,f)
print("saved neighbour list")

# 初始化特征数组
extracted_features = np.ndarray((num_images, BOTTLENECK_TENSOR_SIZE))
graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = create_inception_graph()


# 使用兼容模式创建会话
with tf.compat.v1.Session(graph=graph) as sess:
    # 提取图像特征
    for i, filename in enumerate(neighbor_list):
        try:
            image_data = gfile.GFile(filename, 'rb').read()
            features = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
            extracted_features[i:i+1] = features
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")

        if i % 250 == 0:
            print(f"已处理 {i} 张图像")


# 保存已提取的特征
np.savetxt("saved_features_recom.txt", extracted_features)
print("saved exttracted features")
