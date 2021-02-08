import numpy as np
import pickle

dataset = np.load("lfw_feats_512.npy")

image_files = []
with open("lfw_imagename.txt","r") as file:
    lines = file.readlines()
    for line in lines:
        image_files.append(line)

image_files = np.asarray(image_files)

labels = np.loadtxt("lfw_labels.txt").astype(int)

dict = {
    'features' : dataset,
    'image_files' : image_files,
    'labels' : labels
}

# Save dict into a file
with open("lfw_vgg16.pickle", "wb") as f:
    pickle.dump(dict, f)
