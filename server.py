import os
import urllib
import keras
import numpy as np
import json
import scipy
import datetime
import matplotlib.pyplot as plt
from flask import Flask, request, redirect, flash, url_for, Session
from werkzeug.utils import secure_filename
from solution import create_model, predict
import tensorflow as tf


# Upload settings
UPLOAD_FOLDER = '/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)
ALLOWED_EXTENSIONS = {'csv'}
DECISION_THRESHOLD = 0.5

# Flask settings
sess = Session()
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SESSION_TYPE'] = 'filesystem'
app.secret_key = '07145d5b193b4d3b909f218bccd65be7'

# Training args
with open('params.json', encoding='utf-8') as f:
    params = json.load(f)
    params['batch_size'] = 1

model = create_model(params)
model.load_weights('model.h5')

global graph
graph = tf.get_default_graph()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_predict', methods=['GET', 'POST'])
def upload_file_endpoint():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            target_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(os.path.join(target_file_path))
            global graph
            print("Starting prediction")
            with graph.as_default():
                predictions = predict(in_file=target_file_path, model=model, params=params, current_time=None)
            print("Finished prediction")
            return json.dumps(predictions.to_json())
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''


if __name__ == '__main__':
    app.run(host='0.0.0.0')
