
# coding: utf-8

# In[1]:


import base64
import numpy as np 
import tensorflow as tf 
from io import BytesIO
from flask import Flask, request, jsonify
from keras.models import load_model
from PIL import Image

NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
CAPTCHA_CHARSET = NUMBER   # 验证码字符集
CAPTCHA_LEN = 4            # 验证码长度
CAPTCHA_HEIGHT = 60        # 验证码高度
CAPTCHA_WIDTH = 160        # 验证码宽度

# 训练好的模型文件
MODEL_FILE = './model/train_demo/captcha_adam_binary_crossentropy_bs_100_epochs_10.h5'

def vec2text(vector):
    if not isinstance(vector, np.ndarray):
        vector = np.asarray(vector)
    vector = np.reshape(vector, [CAPTCHA_LEN, -1])
    text = ''
    for item in vector:
        text += CAPTCHA_CHARSET[np.argmax(item)]
    return text

def rgb2gray(image):
    return np.dot(image[...,:3], [0.299,0.587,0.114])

app = Flask(__name__)

# 测试URL
@app.route('/test', methods=['GET','POST'])
def hello_world():
    return 'Hello World!'

# 验证码识别URL
@app.route('/predict', methods=['POST'])
def predict():
    response = {'success':False, 'prediction':'', 'debug':'error'}
    receive_image =False
    if request.method == 'POST':
        if request.files.get('image'): # 获取图像文件
            image = request.files['image'].read()
            receive_image = True 
            response['debug'] = 'get image'
        elif request.files.get_json(): # 获取base64 编码的图像文件
            encode_image = request.get_json()['image']
            image = base64.b64decode(encode_image)
            receive_image = True 
            response['debug'] = 'get json'
        if receive_image:
            image = np.array(Image.open(BytesIO(image)))
            image = rgb2gray(image).reshape(
                1,60,160,1).astype('float32')/255
            with graph.as_default():
                pred = model.predict(image)
            response['prediction'] = response['prediction'] + vec2text(pred)
            response['sucess'] = True
            response['debug'] = 'predicted'
        else:
            response['debug'] = "It's not POST"
        return jsonify(response)
                    
model = load_model(MODEL_FILE)  # 加载模型
graph = tf.get_default_graph()  # 获取TensorFlow数据流图

