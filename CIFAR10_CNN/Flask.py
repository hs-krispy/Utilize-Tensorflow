from flask import Flask, render_template, request
import numpy as np
from urllib import request as rq
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.vgg19 import preprocess_input
result = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

app = Flask(__name__)

@app.route('/')
def render_file():
    return render_template('web.html', res="")

@app.route('/load_image', methods= ['POST', 'GET'])
def load_image():
    model = load_model("VGG16.h5")
    if request.method == 'POST':
        url = request.form['url']
        rq.urlretrieve(url, 'templates/test.jpg')
    img = load_img('templates/test.jpg')
    img = img.resize((32, 32))
    arr = img_to_array(img)
    arr = arr.reshape(1, 32, 32, 3)
    arr = preprocess_input(arr)
    label = model.predict(arr) # 각 클래스 속할 확률
    pred = np.argmax(label) # 확률이 가장 높은 클래스
    return render_template('web.html', res=result[pred], url=url)

if __name__== '__main__':
    app.run(host="127.0.0.1", debug=True)