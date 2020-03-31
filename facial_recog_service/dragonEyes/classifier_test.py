import classifier_svm as clf_svm
import logging
import argparse
import preProcess as pp
import os
import shutil
import modify_image as mi

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('--model-path', type=str, action='store', dest='model_path', default='transferredSkills/faceNet_models/20180402-114759/20180402-114759.pb',
                    help='Path to model protobuf graph')
parser.add_argument('--input-dir', type=str, action='store', dest='input_dir', default='data/custom',
                    help='Input path of data to train on')
parser.add_argument('--batch-size', type=int, action='store', dest='batch_size',
                    help='Input path of data to train on', default=128)
parser.add_argument('--num-threads', type=int, action='store', dest='num_threads', default=16,
                    help='Number of threads to utilize for queue')
parser.add_argument('--num-epochs', type=int, action='store', dest='num_epochs', default=3,
                    help='Path to output trained classifier model')
parser.add_argument('--split-ratio', type=float, action='store', dest='split_ratio', default=0.7,
                    help='Ratio to split train/test dataset')
parser.add_argument('--min-num-images-per-class', type=int, action='store', default=2,
                    dest='min_images_per_class', help='Minimum number of images per class')
parser.add_argument('--classifier-path', type=str, action='store', dest='classifier_path', default='memory/classifier.pkl',
                    help='Path to output trained classifier model')
parser.add_argument('--is-train', action='store_true', dest='is_train', default=False,
                    help='Flag to determine if train or evaluate')

args = parser.parse_args()

if args.is_train:
    clf_svm.main(input_directory=args.input_dir, model_path=args.model_path, classifier_output_path=args.classifier_path,
            batch_size=args.batch_size, num_threads=args.num_threads, num_epochs=args.num_epochs,
            min_images_per_labels=args.min_images_per_class, split_ratio=args.split_ratio, is_train=args.is_train)
else:
    # crop and store in the same subfolder
    '''
    if os.path.exists('tests4_output'):
        shutil.rmtree('tests4_output', ignore_errors=True)
    pp.main('tests/recognition_test/', 'tests4_output/recognition_cropped_test', 200, isGroup=True)
    '''
    # assosciate rect coordinates with cropped image (id)
    target = 'tests4_output/recognition_cropped_test/class_1'
    #mi.main(target)
    # classify
    #result = clf_svm.classify(input_directory='data/test', model_path=args.model_path, classifier_output_path=args.classifier_path,
    #                batch_size=args.batch_size, num_threads=args.num_threads)
    
    folders = [x for x in os.listdir(target) if '_jpg' in x]


    for f in folders:
        f = os.path.join(target, f)
        result = clf_svm.classify(input_directory=f, model_path=args.model_path, classifier_output_path=args.classifier_path,
                    batch_size=args.batch_size, num_threads=args.num_threads)
        #print("####", result)
        mi.main(f, result)


    # plot over main image
    #print(result)
