from app import app
from flask import jsonify, make_response, request, render_template
from darknet import performDetectTraffic
import numpy as np
from cv2 import *
import os

@app.route('/traffic')
def index():
    return render_template('trafficSign.html')

@app.route('/traffic/detection',  methods=['POST'])
def detecting():
    user_name = request.form.get('user-name')
    image_name = request.form.get('image-name')
    image = request.files.get('image').read()

    # convert string data to numpy array
    npimg = np.fromstring(image, np.uint8)
    # convert numpy array to image
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    # get bounding box
    boundingbox = performDetectTraffic(imageContent=img)

    response_body = {
        "user-name": user_name,
        "image-name": image_name,
        "boundingbox": boundingbox
    }

    res = make_response(jsonify(response_body), 201)

    return res
