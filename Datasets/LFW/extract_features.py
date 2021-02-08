from keras.engine import Model
from keras.layers import Input
import numpy as np
from keras_vggface import VGGFace
from keras.preprocessing import image
from keras_vggface import utils
import glob
import cv2
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace

# LOAD MODEL
vgg_features = VGGFace(include_top=False, input_shape=(224, 224, 3), pooling='avg') # pooling: None, avg or max

# LIST FILES
files = sorted(glob.glob('/home/cristianopatricio/Documents/Datasets/LFW/lfwcrop_color/faces/*.ppm'))

print(files)

feats_list = []
# CALCULATE FEATURES
for imagepath in files:

    print(imagepath)
    im = cv2.imread(imagepath)
    im = cv2.resize(im, (224,224))
    im_np = np.expand_dims(im, axis=0)
    im_np = im_np.astype('float32')
    im_preproc = utils.preprocess_input(im_np, version=1)

    feats = vgg_features.predict(im_preproc)
    feats_list.append(feats[0])

    if len(feats_list)%1000 == 0:
        np.save('lfw_feats_512.npy', feats_list)

np.save('lfw_feats_512.npy', feats_list)

