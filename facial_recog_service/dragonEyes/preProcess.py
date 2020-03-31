
import argparse
import glob
import logging
import os
import time
import uuid
import cv2
import shutil
import csv

from dragonEyes.align_dlib import AlignDlib

logger = logging.getLogger(__name__)

align_dlib = AlignDlib(os.path.join(os.path.dirname(__file__), 'transferredSkills/dlib_models/shape_predictor_68_face_landmarks.dat'))


def main(image_path, output_dir, crop_dim, isGroup = False):
    start_time = time.time()
    #pool = mp.Pool(processes=mp.cpu_count())

    input_dir = os.path.dirname(image_path)
    output_dir = os.path.dirname(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #image_paths = glob.glob(os.path.join(input_dir, '**/*.jpg'))
    #image_paths = glob.glob(input_dir + '/*.jpg')

    if isGroup:
        #for image_path in image_paths:
        #image_output_dir = os.path.join(output_dir, os.path.join(os.path.basename(os.path.dirname(image_path)), os.path.basename(image_path).replace('.', '_')))
        image_output_dir = os.path.join(output_dir, os.path.basename(image_path).replace('.jpg', '_jpg'))
        #print('****', image_output_dir)
        if not os.path.exists(image_output_dir):
            os.makedirs(image_output_dir)

        #for index, image_path in enumerate(image_paths):
        image_output_dir = os.path.join(os.path.join(output_dir, os.path.basename(os.path.dirname(image_path)), os.path.basename(image_path).replace('.', '_')))
        #output_path = os.path.join(image_output_dir, os.path.basename(image_path))
        output_path = os.path.join(output_dir, os.path.basename(image_path).replace('.jpg', '_jpg'))
        
        #pool.apply_async(preprocess_image, (image_path, output_path, crop_dim, True))
        preprocess_image(image_path, output_path, crop_dim, True)

    else:
        for image_dir in os.listdir(input_dir):
            image_output_dir = os.path.join(output_dir, os.path.basename(os.path.basename(image_dir)))
            if not os.path.exists(image_output_dir):
                os.makedirs(image_output_dir)

        #image_paths = glob.glob(os.path.join(input_dir, '**/*.jpg'))
        #for index, image_path in enumerate(image_paths):
        
        image_output_dir = os.path.join(output_dir, os.path.basename(os.path.dirname(image_path)))
        output_path = os.path.join(image_output_dir, os.path.basename(image_path))
        
        #pool.apply_async(preprocess_image, (image_path, output_path, crop_dim, True))
        preprocess_image(image_path, output_path, crop_dim, True)


    #pool.close()
    #pool.join()
    logger.info('Completed in {} seconds'.format(time.time() - start_time))



def preprocess_image(input_path, output_path, crop_dim, isGroup = False):
    """
    Detect face, align and crop :param input_path. Write output to :param output_path
    :param input_path: Path to input image
    :param output_path: Path to write processed image
    :param crop_dim: dimensions to crop image to
    """
    csv_dict = {}
    images = _process_image(input_path, crop_dim)
    #print(bb)
    if images is not None:
        logger.debug('Writing processed file: {}'.format(output_path))
        i = 1
        for image in images:
            #_output_path = os.path.join(os.path.dirname(output_path), str(i) + '__' + str(uuid.uuid4()).replace('-', '') + '.jpg')
            _output_path = os.path.join(output_path, str(i) + '__' + str(uuid.uuid4()).replace('-', '') + '.jpg')
            csv_dict[os.path.basename(_output_path)] = str(align_dlib._getBoxCoordinates(image[1])).replace('(', '').replace(')', '').replace(' ', '')
            cv2.imwrite(_output_path , image[0])
            i += 1

        # copy original file to destination
        #shutil.copy(input_path, os.path.dirname(os.path.dirname(output_path)))
        
        if isGroup:
            csv_path = os.path.join(output_path, os.path.basename(output_path).replace('_jpg', '_jpg.csv'))
            #print('***********', csv_path)
            with open (csv_path, 'w', newline='') as f:
                g = csv.writer(f)
                columnTitleRow = ["segment", "bb", 'name_map', 'student_id', 'name', 'accuracy']
                g.writerow(columnTitleRow)
                for key in csv_dict.keys():
                    segment = key
                    bb = csv_dict[key]
                    row = [segment, bb, '-1', '-1', 'DragonSlayer', '-1']
                    g.writerow(row)
    else:
        logger.warning("Skipping filename: {}".format(input_path))


def _process_image(filename, crop_dim):
    image = None
    aligned_image = None

    image = _buffer_image(filename)

    if image is not None:
        aligned_image = align_images(image, crop_dim)
    else:
        raise IOError('Error buffering image: {}'.format(filename))

    return aligned_image


def _buffer_image(filename):
    logger.debug('Reading image: {}'.format(filename))
    image = cv2.imread(filename, )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _align_image(image, bb, crop_dim):
    aligned = align_dlib.align(crop_dim, image, bb, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
    if aligned is not None:
        aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    return aligned

def align_images(image, crop_dim):
    bbs = align_dlib.getAllFaceBoundingBoxes(image)
    return zip([_align_image(image, bb, crop_dim) for bb in bbs], bbs)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--input-dir', type=str, action='store', default='lfw', dest='input_dir')
    parser.add_argument('--output-dir', type=str, action='store', default='output', dest='output_dir')
    parser.add_argument('--crop-dim', type=int, action='store', default=100, dest='crop_dim',
                        help='Size to crop images to')

    args = parser.parse_args()

    if os.path.exists('tests3_output'):
        shutil.rmtree('tests3_output')

    #main(args.input_dir, args.output_dir, args.crop_dim)
    main('tests/crop_test', 'tests3_output/face_segments', 200, isGroup=True)

    #for recognition testing
    main('tests/recognition_train', 'tests3_output/recognition_cropped', 200)
    main('tests/recognition_test', 'tests3_output/recognition_cropped_test', 200, isGroup=True)