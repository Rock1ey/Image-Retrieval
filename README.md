# Introduction
    本次作业采用Anaconda创建虚拟环境来安装程序所需要的依赖，运行过程如下：
    1.运行image_vectorizer.py，它将每个数据通过 Inception-v3 模型并收集瓶颈层向量并存储在磁盘中；
    2.运行rest-server.py启动服务器，通过Flask实现UI的REST服务；
    3.启动服务器后，通过 URL（例如 0.0.0.1:5000）访问 UI。上传任意文件并查看k张相似图像。

# Requirements
    ·Anaconda
    ·必要的包：
      flask pyqt numpy tensorflow flask-httpauth scipy imageio matplotlib scikit-learn

# How To Run?
    1.创建虚拟环境
    打开 Anaconda Prompt 或终端，运行以下命令创建一个新的虚拟环境（假设命名为 image_search_env）：
    # sh
        conda create -n image_search_env python=3.8
    2.激活虚拟环境
    # sh
        conda activate image_search_env
    3.安装必要的软件包
    在激活的虚拟环境中，运行以下命令安装所需的软件包：
    # sh
        conda install flask pyqt numpy tensorflow flask-httpauth scipy imageio matplotlib scikit-learn
