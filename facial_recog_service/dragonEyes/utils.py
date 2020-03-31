from PIL import Image
import numpy as np
import dlib
import face_recognition_models

#common objects to be used in differnt functions
faceDetectorObject = dlib.get_frontal_face_detector()
cnn_face_detection_model = face_recognition_models.cnn_face_detector_model_location()
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model)

predictor_68_point_model = face_recognition_models.pose_predictor_model_location()
pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)

predictor_5_point_model = face_recognition_models.pose_predictor_five_point_model_location()
pose_predictor_5_point = dlib.shape_predictor(predictor_5_point_model)

face_recognition_model = face_recognition_models.face_recognition_model_location()
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)

def loadImage(filePath, mode='RGB'):
    """
        to load a given image in numpy array
        TODO: optimize this function
    """
    """
    f = Image.open(filePath)
    wpercent = (basewidth/float(f.size[0]))
    hsize = int((float(f.size[1])*float(wpercent)))
    f = f.resize((basewidth,hsize), Image.ANTIALIAS)
    f = f.convert(mode)
    """
    f = Image.open(filePath)
    f = f.convert(mode)
    return np.array(f)

# utils for getFaceBound

def _boxToRect(css):
    """
    tuple to dlib rect object
    """
    return dlib.rectangle(css[3], css[0], css[1], css[2])

def _getBoxCoordinates(box):
    return box.top(), box.right(), box.bottom(), box.left()

def trimBoxes(box, image_shape):
    return max(box[0], 0), min(box[1], image_shape[1]), min(box[2], image_shape[0]), max(box[3], 0)

def faceDistance(face_encodings, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.
    """
    if len(face_encodings) == 0:
        return np.empty((0))

    return np.linalg.norm(face_encodings - face_to_compare, axis=1)

def faceBoundary(img, upsample_count=1, model="hog"):
    """
    returns array of boxes for faces in a given image
    upsample_count: more the count more the sampling of image will be done to find faces
    """
    if model == "cnn":
        # TODO: change the name
        return cnn_face_detector(img, upsample_count)
    else:
        return faceDetectorObject(img, upsample_count)

def getFaceBounds(img, upsample_count=1, model="hog"):
    """
    returns all the face boundary in an images
    main functions to get face boundary
    """
    return [trimBoxes(_getBoxCoordinates(face), img.shape) for face in faceBoundary(img, upsample_count, model)]
    
# TODO: downscale a given image and make it configurable

def _facePoints(faceImg, faceLocs=None, model="large"):
    if faceLocs is None:
        faceLocs = faceBoundary(faceImg)
    else:
        faceLocs = [_boxToRect(faceLoc) for faceLoc in faceLocs]

    pose_predictor = pose_predictor_68_point

    if model == "small":
        pose_predictor = pose_predictor_5_point

    return [pose_predictor(faceImg, faceLoc) for faceLoc in faceLocs]


def facePoints(faceImg, faceLocs=None):
    """
    returns face features (lips, eyes etc) given an image
    """
    features = _facePoints(faceImg, faceLocs)
    features_as_tuples = [[(p.x, p.y) for p in feature.parts()] for feature in features]

    # Cite: For a definition of each point index, see https://cdn-images-1.medium.com/max/1600/1*AbEg31EgkbXSQehuNJBlWg.png
    return [{
        "chin": points[0:17],
        "left_eyebrow": points[17:22],
        "right_eyebrow": points[22:27],
        "nose_bridge": points[27:31],
        "nose_tip": points[31:36],
        "left_eye": points[36:42],
        "right_eye": points[42:48],
        "top_lip": points[48:55] + [points[64]] + [points[63]] + [points[62]] + [points[61]] + [points[60]],
        "bottom_lip": points[54:60] + [points[48]] + [points[60]] + [points[67]] + [points[66]] + [points[65]] + [points[64]]
    } for points in features_as_tuples]


def faceEncodings(faceImg, knownFaceLocs=None, resampleCount=1):
    """
    return the 128-dimension face encoding given an image

    knownFaceLocs: the bounding boxes of each face if you already know them.
    resampleCount: How many times to re-sample the face when calculating encoding. Higher is more accurate, but slower 
    """
    raw_landmarks = _facePoints(faceImg, knownFaceLocs, model="small")
    return [np.array(face_encoder.compute_face_descriptor(faceImg, raw_landmark_set, resampleCount)) for raw_landmark_set in raw_landmarks]

def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    """
    Compare a list of face encodings against a candidate encoding to see if they match.
    """
    return list(faceDistance(known_face_encodings, face_encoding_to_check) <= tolerance)