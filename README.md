# Introduction
    本次作业采用Anaconda创建虚拟环境来安装程序所需要的依赖，运行过程如下：
    1.运行image_vectorizer.py，它将每个数据通过 Inception-v3 模型并收集瓶颈层向量并存储在磁盘中；
    2.运行rest-server.py启动服务器，通过Flask实现UI的REST服务；
    3.启动服务器后，通过 URL（例如 0.0.0.1:5000）访问 UI。上传任意文件并查看k张相似图像。

# Requirements
    ├── ReadMe.md           // 帮助文档
    
    ├── AutoCreateDDS.py    // 合成DDS的 python脚本文件
    
    ├── DDScore             // DDS核心文件库，包含各版本的include、src、lib文件夹，方便合并
    
    │   ├── include_src     // 包含各版本的include、src文件夹
    
    │       ├── V1.0
    
    │           ├── include
    
    │           └── src
    
    └── temp                // 存放待合并的服务的服务文件夹
                        
原文链接：https://blog.csdn.net/qq_25662827/article/details/124440992

# How To Run?
