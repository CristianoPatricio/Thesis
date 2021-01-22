import numpy as np
import pickle

dataset = np.load("/home/cristianopatricio/Documents/Datasets/Celeb-A/celeba_feats_512.npy")

image_files = []
with open("/home/cristianopatricio/Documents/Datasets/Celeb-A/data/celeba_imagename.txt","r") as file:
    lines = file.readlines()
    path = "/home/cristianopatricio/Documents/Datasets/Celeb-A/img_align_celeba/"
    for line in lines:
        modified_line = path + str(line[-11:])
        image_files.append(modified_line)

image_files = np.asarray(image_files)

labels = []
with open("/home/cristianopatricio/Documents/Datasets/Celeb-A/data/identity_CelebA.txt","r") as file:
    lines = file.readlines()
    for line in lines:
        line = line.rstrip().split()
        labels.append(int(line[1])-1)

labels = np.asarray(labels)

dict = {
    'features' : dataset,
    'image_files' : image_files,
    'labels' : labels
}

# Save dict into a file
with open("vgg16.pickle", "wb") as f:
    pickle.dump(dict, f)
