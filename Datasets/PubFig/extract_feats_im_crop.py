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
                    help='Directory path containing cropped images.')


if __name__ == '__main__':
    args = parser.parse_args()
    dir_path = args.images_dir
    output_filename = 'pubfig_feats_vgg_512_test.npy'

    # LOAD MODEL
    vgg_features = VGGFace(include_top=False, input_shape=(224, 224, 3), pooling='avg')  # pooling: None, avg or max

    # LIST FILES
    files = sorted(glob.glob(dir_path+'*.jpg'))

    feats_list = []
    # CALCULATE FEATURES
    for i, image_path in enumerate(files):
        print("[INFO]: Extracting {0} ...".format(image_path))
        image_name = os.path.basename(image_path)[:-4]
        im = image.load_img(image_path)
        im = cv2.resize(np.uint8(im), (224, 224))
        im_np = np.expand_dims(im, axis=0)
        im_np = im_np.astype('float32')
        im_preproc = utils.preprocess_input(im_np, version=1)

        feats = vgg_features.predict(im_preproc)
        feats_list.append(feats[0])

        if len(feats_list) % 1000 == 0:
            np.save(output_filename, feats_list)

    np.save(output_filename, feats_list)
