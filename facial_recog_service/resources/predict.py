from flask_restful import Resource, reqparse
from flask import Flask, request, Response
import pandas as pd
import os
import dragonEyes as eyes
import shutil
import cv2
import numpy as np
from PIL import Image, ImageDraw
import uuid
import requests

data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
model_path = 'dragonEyes/transferredSkills/faceNet_models/20180402-114759/20180402-114759.pb'
classifier_path = 'dragonEyes/memory/classifier.pkl'

class Predict(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument("data",
                        type=str,
                        required=True,
                        help="image data to parse"
                        )

    def get(self):
        # returns all the available models
        pass
    
    def post(self):
        # runs a model based on the input and returns the predictions
        #data = Predict.parser.parse_args()
        r = request
        print(r.headers)
        nparr = np.fromstring(r.data, np.uint8)
        #print(r.data.name)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        pil_image = Image.fromarray(img)
        file_name = './test/trtr' + str(uuid.uuid1()) + '.jpg'
        pil_image.save(file_name, "JPEG")
        do_predict(file_name)
        img2 = cv2.imread(file_name.replace('.jpg', '_processed.jpg'))
        _, img_encoded2 = cv2.imencode('.jpg', img2)
        content_type = 'image/jpeg'
        headers = {'content-type': content_type}
        requests.post('http://localhost:5002/frames_to_video', data=img_encoded2.tostring(), headers=headers)
        return 1


# ----------------HELPER FUNCIONS------------------------------------------
def crop_face(image_url):
    #print(os.path.dirname(image_url))
    eyes.find_faces(image_url, image_url, 200, isGroup=True)

def clear_folder(image_url):
    if os.path.exists(image_url.replace('.jpg', '_jpg')):
        shutil.rmtree(image_url.replace('.jpg', '_jpg'), ignore_errors=True)

def do_predict(img_url):
    try:
        print(img_url)
        if os.path.exists(img_url.replace(".jpg", "_jpg")):
            return

        #img_url = request.headers["img_url"]
        clear_folder(img_url)
        crop_face(img_url)
        folder = img_url.replace('.jpg', '_jpg')

        result = eyes.clf_svm(input_directory=folder, model_path=model_path, classifier_output_path=classifier_path,
                                batch_size=128, num_threads=16)

        eyes.visualize(folder, result)
        head, tail = os.path.split(folder)
        index = ((tail).find("_"))
        studentID = tail[:index]
        print(studentID)
        #uploadStudentSectionImage(studentID)
    except ValueError:
        return 'error'
    return 'error'