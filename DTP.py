# from tensorflow.python.keras.layers import Dropout, Flatten, Dense
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# import numpy as np
# import tensorflow as tf
from flask import Flask, request, render_template
import os
SIZE = 224

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

# def resize_image(img):
#     img = tf.image.resize(img, (SIZE, SIZE))
#     img = tf.cast(img, tf.float32)
#     img = img / 255.0
#     return img
#
#
# base_layers = tf.keras.applications.MobileNetV2(input_shape=(SIZE, SIZE, 3), include_top=False)
# base_layers.trainable = False
#
# model = tf.keras.Sequential([
#     base_layers,
#     Flatten(),
#     Dense(256, activation="relu"),
#     Dropout(0.5),
#     Dense(128, activation="relu"),
#     Dropout(0.5),
#     Dense(64, activation="relu"),
#     Dropout(0.5),
#     Dense(4, activation="sigmoid"),
# ])
#
# model.load_weights('model_test/test.h5')
#
# test = ['dents', 'headlights', 'scratches', 'totals']

@app.route('/predict',methods=['POST'])
def predict():
    name = request.files['img']
    name.filename = 'asd.jpg'
    name.save(name.filename)

    # img = load_img('asd.jpg')
    # img_array = img_to_array(img)
    # img_resized = resize_image(img_array)
    # img_expended = np.expand_dims(img_resized, axis=0)
    # prediction = model.predict(img_expended)
    # predicted_label = np.argmax(prediction)
    # pred = np.max(prediction)
    # pr = test[predicted_label]
    #
    # test_2 = int(pred * 100)

    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'asd.jpg')
    os.remove(path)


    return render_template('index.html', pred=f'{name}-{name.filename}')



if __name__ == '__main__':
    app.run(debug=True)
