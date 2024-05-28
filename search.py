################################################################################################################################
# 这个函数实现了图像搜索/检索功能。
# 输入: 上传图像的位置，提取的特征向量
################################################################################################################################
import random
import tensorflow.compat.v1 as tf  # type: ignore # 使用兼容模式引入 TensorFlow 1.x
import numpy as np
import imageio
import os
import scipy.io
import time
from datetime import datetime
from scipy import ndimage
from scipy.spatial.distance import cosine
from sklearn.neighbors import NearestNeighbors
import pickle
from PIL import Image
import gc
from tempfile import TemporaryFile
from tensorflow.python.platform import gfile

# 禁用 TensorFlow 2.x 行为
tf.compat.v1.disable_v2_behavior()

BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

def get_top_k_similar(image_data, pred, pred_final, k):
    '''
    获取k个最接近检索图片的索引
    '''
    print("total data", len(pred))
    print(image_data.shape)
    os.makedirs('static/result', exist_ok=True)  # 使用 os.makedirs 创建目录，如果目录已存在则忽略
    
    top_k_ind = np.argsort([cosine(image_data, pred_row) for pred_row in pred])[:k]
    print(top_k_ind)
    
    for i, neighbor in enumerate(top_k_ind):
        if neighbor >= len(pred_final):
            print(f"Warning: neighbor index {neighbor} out of range")
            continue
        print("Attempting to read image at:", pred_final[neighbor])
        image = imageio.imread(pred_final[neighbor])
        name = pred_final[neighbor]
        tokens = name.split("\\")
        img_name = tokens[-1]
        print(img_name)
        name = os.path.join('static', 'result', img_name)
        imageio.imsave(name, image)

def create_inception_graph():
    """从保存的GraphDef文件创建图，并返回一个图对象。

    返回:
        包含训练过的Inception网络的图，以及我们将操作的各种张量。
    """
    with tf.Session() as sess:
        # 构建模型文件的路径
        model_filename = os.path.join('imagenet', 'classify_image_graph_def.pb')
        
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
                tf.import_graph_def(graph_def, name='', return_elements=[
                    BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
                    RESIZED_INPUT_TENSOR_NAME]))
    return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor

def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            bottleneck_tensor):
    bottleneck_values = sess.run(
            bottleneck_tensor,
            {image_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values

def recommend(imagePath, extracted_features):
    tf.compat.v1.reset_default_graph()  # 使用 TensorFlow 1.x 的 reset_default_graph

    config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

    with tf.Session(config=config) as sess:
        graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = create_inception_graph()
        image_data = gfile.FastGFile(imagePath, 'rb').read()
        features = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)    

        with open('neighbor_list_recom.pickle', 'rb') as f:
            neighbor_list = pickle.load(f)
        print("loaded images")
        print("Neighbor list length:", len(neighbor_list))
        get_top_k_similar(features, extracted_features, neighbor_list, k=9)
