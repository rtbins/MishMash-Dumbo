
import os
from dragonEyes.align_dlib import AlignDlib
import csv
import glob
import cv2
from PIL import Image, ImageDraw
import pandas as pd
from random import shuffle

align_dlib = AlignDlib(os.path.join(os.path.dirname(__file__), 'transferredSkills/dlib_models/shape_predictor_68_face_landmarks.dat'))

def draw_on_image(image, out_loc, boxes, face_landmarks_list = None):

    #image = utils.loadImage(image_path)

    colors = ['blue', 'red', '#7FFF00', 'yellow', '#BF3EFF', '#121212', '#FF69B4', '#FFA54F']

    # Find all facial features in all the faces in the image
    #face_landmarks_list = utils.facePoints(image)
    #boxes = utils.getFaceBounds(image, 2, 'hog')


    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image, 'RGBA')

    if face_landmarks_list:
        for i in range(len(face_landmarks_list)):
            color = colors[i%len(colors)]
            face_landmarks = face_landmarks_list[i]

            # Make the eyebrows into a nightmare
            d.point(face_landmarks['left_eyebrow'], fill=color)
            d.point(face_landmarks['right_eyebrow'], fill=color)
            
            d.point(face_landmarks['chin'], fill=color)

            # lips
            d.point(face_landmarks['top_lip'], fill=color)
            d.point(face_landmarks['bottom_lip'], fill=color)
            '''
            d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
            d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
            d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
            d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)
            '''
            d.point(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=color)
            d.point(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=color)

    #print('deferfr', boxes)
    for i in range(len(boxes)):
        #print(boxes[i])
        box = boxes[i][1]
        #print(box[3], box[0]), (box[1], box[2])
        color = colors[i%len(colors)]
        d.rectangle(((box[3], box[0]), (box[1], box[2])), outline = color)
        d.text((box[3], box[2]), text=boxes[i][0], fill = color)
        #d.text((box[3], box[2]), text=str(boxes[i][2]), fill = color)

    
    pil_image.save(out_loc, "JPEG")

def _buffer_image(filename):
    image = cv2.imread(filename, )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image 

def get_image_details(path):
    result = []
    header = []
    isHeader = True
    with open(path) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        dict_list = []
        for row in reader:
            if isHeader:
                header = row
                isHeader = False
            else:
                result.append(dict(zip(header, row)))
    return result

def main(img_dir, result):

    #for root, dirs, files in os.walk(img_dir):
    
    #for file in glob.glob(os.path.join(img_dir,"**/*.csv")):

    file = os.path.join(img_dir, os.path.join(os.path.basename(img_dir).replace('_jpg', '_jpg.csv')))
    df_f = pd.read_csv(file)
    #print(file)
    boxes = []
    #print(file)
    for d in get_image_details(file):
        if len(list(d.keys())) > 0:
            parent_file = os.path.dirname(file).replace('_jpg', '.jpg')
            #print(d)
            bb = [int(s) for s in d['bb'].replace('L','').split(',') if s.isdigit()]
            r = result[d['segment']]

            df_f.loc[df_f['segment']==d['segment'], 'accuracy'] = r.accuracy
            df_f.loc[df_f['segment']==d['segment'], 'name'] = r.person_name

            boxes.append([r.person_name, bb, r.accuracy])

    #print(os.path.dirname(file).replace('_jpg', '.jpg'))
    out_path = os.path.dirname(file).replace('_jpg', '_processed.jpg')
    #print('o',out_path)
    _image = _buffer_image(os.path.dirname(file).replace('_jpg', '.jpg'))
    draw_on_image(_image, out_path, boxes)
    df_f.to_csv(file, index=False)


if __name__ == '__main__':
    import os
    import glob
    import cv2
    from align_dlib import AlignDlib

    align_dlib = AlignDlib(os.path.join(os.path.dirname(__file__), 'transferredSkills/dlib_models/shape_predictor_68_face_landmarks.dat'))

    def _buffer_image(filename):
        image = cv2.imread(filename, )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image 

    out_dir = 'tests_output/draw_test'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


    image_paths = glob.glob(os.path.join('tests/crop_test', '**/*.jpg'))
    for image in image_paths:
        _image = _buffer_image(image)
        boxes = [ align_dlib._getBoxCoordinates(b) for b in align_dlib.getAllFaceBoundingBoxes(_image)]
        landmarks = align_dlib.facePoints( _image, boxes)

        out_path = os.path.join(out_dir, os.path.basename(image)).replace('.jpg', '') + "_processed.jpg"
        draw_on_image(_image, out_path, boxes, landmarks)