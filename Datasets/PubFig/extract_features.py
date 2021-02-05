"""
Python script to extract image features of the PubFig dataset.
"Attribute and Simile Classifiers for Face Verification,"
Neeraj Kumar, Alexander C. Berg, Peter N. Belhumeur, and Shree K. Nayar,
International Conference on Computer Vision (ICCV), 2009.

Author: Cristiano Patrício
E-mail: cristiano.patricio@ubi.pt
University of Beira Interior, Portugal
"""

from keras.engine import Model
from keras.layers import Input
import numpy as np
from keras_vggface import VGGFace
from keras.preprocessing import image
from keras_vggface import utils
import glob
import cv2
import os
import re
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
import argparse

# Args
parser = argparse.ArgumentParser(description='Extract PubFig Images Features')
parser.add_argument('--images-dir', type=str, default='',
                    help='Directory path containing dev/eval images.')
parser.add_argument('--url-file-dir', type=str, default='',
                    help='Directory path containing dev_urls.txt / eval_urls.txt file.')


##########################################################################
# AUXILIARY FUNCTIONS
##########################################################################


def get_image_bb(file):
    name_to_rect = dict()
    with open(file, 'r') as f:
        lines = f.readlines()
        for l in lines[2:]:
            line = re.split("\t", l)
            # rect is in the 3 position
            rect = line[3].split(",")
            x0 = int(rect[0])
            y0 = int(rect[1])
            x1 = int(rect[2])
            y1 = int(rect[3])
            arr_rect = np.array([x0, y0, x1, y1])

            # image name line 0 + 1
            pre_name = line[0].replace(" ", "_")
            id = int(line[1])

            if id < 10:
                name = pre_name + str("_") + str('000') + str(id)
            elif 10 <= id < 100:
                name = pre_name + str("_") + str('00') + str(id)
            elif 100 <= id < 1000:
                name = pre_name + str("_") + str('0') + str(id)
            else:
                name = pre_name + str("_") + str(id)

            name_to_rect[name] = arr_rect

    return name_to_rect


if __name__ == '__main__':
    args = parser.parse_args()
    dir_path = args.images_dir
    file_path = args.url_file_dir
    output_filename = 'pubfig_feats_dev_512.npy' if "dev" in file_path else 'pubfig_feats_eval_512.npy'

    # LOAD MODEL
    vgg_features = VGGFace(include_top=False, input_shape=(224, 224, 3), pooling='avg')  # pooling: None, avg or max

    # LIST FILES
    files = sorted(glob.glob(dir_path+'*.jpg'))

    # BB Coordinates
    get_rect = get_image_bb(file=file_path)

    feats_list = []
    # CALCULATE FEATURES
    for i, image_path in enumerate(files):
        print("[INFO]: Extracting {0} ...".format(image_path))
        image_name = os.path.basename(image_path)[:-4]
        im = image.load_img(image_path)
        bb = get_rect[image_name]
        im = im.crop(bb)
        im = cv2.resize(np.uint8(im), (224, 224))
        im_np = np.expand_dims(im, axis=0)
        im_np = im_np.astype('float32')
        im_preproc = utils.preprocess_input(im_np, version=1)

        feats = vgg_features.predict(im_preproc)
        feats_list.append(feats[0])

        if len(feats_list) % 1000 == 0:
            np.save(output_filename, feats_list)

    np.save(output_filename, feats_list)
