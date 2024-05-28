#!flask/bin/python
################################################################################################################################
# 这个文件实现了REST层。它使用Flask微框架进行服务器实现。来自前端的调用以JSON格式到达这里，并被分支到各个项目。在此文件中还进行了基本级别的验证。
################################################################################################################################
from flask import Flask, jsonify, request, redirect, render_template, url_for
from werkzeug.utils import secure_filename
import os
import shutil 
import numpy as np
from search import recommend
from tensorflow.python.platform import gfile

UPLOAD_FOLDER = 'uploads'  # 上传文件夹
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}  # 允许的文件扩展名

app = Flask(__name__, static_url_path = "")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 加载用于图像检索的提取特征向量
extracted_features = np.zeros((2955, 2048), dtype=np.float32)
with open('saved_features_recom.txt') as f:
    for i, line in enumerate(f):
        extracted_features[i, :] = line.split()
print("加载提取的特征完成")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/imgUpload', methods=['POST'])
def upload_img():
    print("图像上传")
    result = 'static/result'
    if not gfile.Exists(result):
        os.mkdir(result)
    shutil.rmtree(result)

    if 'file' not in request.files:
        print('没有文件部分')
        return redirect(request.url)

    file = request.files['file']
    print(file.filename)
    if file.filename == '':
        print('没有选择文件')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        recommend(filepath, extracted_features)
        os.remove(filepath)

        image_path = 'static/result'
        image_list = [os.path.join(image_path, img) for img in os.listdir(image_path) if not img.startswith('.')]
        images = {f'image{i}': url_for('static', filename=f'result/{os.path.basename(img)}') for i, img in enumerate(image_list)}

        return jsonify(images)
    else:
        print('文件类型不被允许')
        return redirect(request.url)
    
@app.route('/refineSearch', methods=['POST'])
def refine_search():
    data = request.get_json()
    image_src = data.get('imageSrc')
    if not image_src:
        return jsonify({'error': 'No image selected for refinement'}), 400

    image_filename = os.path.basename(image_src)
    image_path = os.path.join('static', 'result', image_filename)

    if not os.path.exists(image_path):
        return jsonify({'error': 'Selected image not found'}), 404

    recommend(image_path, extracted_features)
    
    image_path = 'static/result'
    image_list = [os.path.join(image_path, img) for img in os.listdir(image_path) if not img.startswith('.')]
    images = {f'image{i}': url_for('static', filename=f'result/{os.path.basename(img)}') for i, img in enumerate(image_list)}

    return jsonify(images)

@app.route("/")
def main():
    return render_template("main.html")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
