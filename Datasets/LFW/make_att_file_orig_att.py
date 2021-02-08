import numpy as np
import math
import re
import pickle
from sklearn import preprocessing
from scipy import stats

#################################################################
# Auxiliary Functions
#################################################################


lfw_attributes_names = ["Male", "Asian", "White", "Black", "Baby", "Child", "Youth", "Middle Aged", "Senior",
                        "Black Hair", "Blond Hair", "Brown Hair", "Bald", "No Eyewear", "Eyeglasses", "Sunglasses",
                        "Mustache", "Smiling", "Frowning", "Chubby", "Blurry", "Harsh Lighting", "Flash",
                        "Soft Lighting", "Outdoor", "Curly Hair", "Wavy Hair", "Straight Hair", "Receding Hairline",
                        "Bangs", "Sideburns", "Fully Visible Forehead", "Partially Visible Forehead",
                        "Obstructed Forehead", "Bushy Eyebrows", "Arched Eyebrows", "Narrow Eyes", "Eyes Open",
                        "Big Nose", "Pointy Nose", "Big Lips", "Mouth Closed", "Mouth Slightly Open", "Mouth Wide Open",
                        "Teeth Not Visible", "No Beard", "Goatee", "Round Jaw", "Double Chin", "Wearing Hat",
                        "Oval Face", "Square Face", "Round Face", "Color Photo", "Posed Photo", "Attractive Man",
                        "Attractive Woman", "Indian", "Gray Hair", "Bags Under Eyes", "Heavy Makeup", "Rosy Cheeks",
                        "Shiny Skin", "Pale Skin", "5 o' Clock Shadow", "Strong Nose-Mouth Lines", "Wearing Lipstick",
                        "Flushed Face", "High Cheekbones", "Brown Eyes", "Wearing Earrings", "Wearing Necktie",
                        "Wearing Necklace"]


def image_name2image_id(image_name):
    return int(image_name[:-4])


def image_2_id():
    dict = {}
    with open("lfw_identity.txt", 'r') as file:
        lines = file.readlines()
        for idx, l in enumerate(lines):
            line = l.split()
            image_name = str(line[0])
            image_id = idx
            dict[image_name] = image_id

    return dict


def image2id_abbrev():
    dict = {}
    with open(lfw_identity_file, 'r') as file:
        lines = file.readlines()
        for idx, l in enumerate(lines):
            line = l.split()
            image_name = str(line[0][:-4])
            image_id = idx
            dict[image_name] = image_id

    return dict


def load_lfw_identities(lfw_identities_files, images_ids):
    """"
    input: celeba_identities_file - path to the file containing CELEB-A IDs

        identity_CelebA.txt

        image_name_1 person_id_1
        ...
        image_name_n person_id_n


    output: identity_info - dictionary of the list image names per id

        identity_info[person_id] -> (image_name_1, ..., image_name_n)
        image_info[image_id] -> person_id
    """

    identity_info = dict()
    image_info = dict()
    image2id = image2id_abbrev()
    images = []
    with open(lfw_identities_files) as identities:
        lines = identities.readlines()
        for identity in lines:
            identity = identity.rstrip().lstrip().split()
            # we have 2 infos per line, image name and identity id
            if len(identity) != 2:
                continue
            image_name = identity[0]
            image_id = image2id[image_name[:-4]]
            identity_id = int(identity[1])

            if identity_id not in identity_info:
                identity_info[identity_id] = []

            if image_id in images_ids:
                images.append(image_id)
                identity_info[identity_id].append(image_name)
                image_info[image_id] = identity_id

    return identity_info, image_info, images


def get_index_from_samples(samples, lfw_identity_file):
    idxs = []
    with open(lfw_identity_file, "r") as identities:
        lines = identities.readlines()
        for idx, line in enumerate(lines):
            line = line.rstrip().split()
            if line[0] in samples:
                idxs.append(idx)

    idxs = np.asarray(idxs)

    return idxs


def get_index_from_samples_v2(samples):
    image2id = image_2_id()
    idxs = []
    for sample in samples:
        idxs.append(image2id[sample])

    return idxs


def get_attr_per_samples(attr_path, samples):
    attr_list = []
    with open(attr_path, "r") as attr_file:
        lines = attr_file.readlines()
        for line in lines[2:]:
            values = line.rstrip().split()
            if values[0] in samples:
                attr_list.append([int(v) for v in values[1:]])

    attr_list = np.asarray(attr_list)
    return attr_list


def get_id_by_name(image_name):
    image2id = image2id_abbrev()
    return image2id[image_name]


def load_lfw_attrs(lfw_atts_file):
    """"
    input: lfw_atts_file - path to the file containing LFW attributes

        LFW_attributes.txt
        [header]
        att_names
        person imagenum att_1 ... att_73


    output: identity_info - dictionary of the bb names per image name

        att_info[image_id] -> (att_n_1, att_n_2, ..., att_n_40)

    """

    att_info = dict()
    image2id = image2id_abbrev()
    with open(lfw_atts_file, 'r') as atributes_file:
        lines = atributes_file.readlines()
        assert (len(lines) > 3)

        # second line is the header
        names = []
        images = []
        for line in lines[2:]:

            values = re.split("\t", line)
            name = values[0].replace(" ", "_")
            names.append(name)

            image_num = int(values[1])
            if image_num < 10:
                name_img = name+str('_000')+str(image_num)
            elif 10 <= image_num < 100:
                name_img = name + str('_00') + str(image_num)
            elif 100 <= image_num < 1000:
                name_img = name + str('_0') + str(image_num)
            elif image_num >= 1000:
                name_img = name + str('_') + str(image_num)


            image_id = image2id[name_img]


            attibutes_arr = []
            for i in range(0, len(lfw_attributes_names)):
                attr = lfw_attributes_names[i]
                value = values[i + 2]

                attibutes_arr.append(float(value))

            images.append(image_id)
            att_info[image_id] = np.array(attibutes_arr)

    # print("found ", len(celeb_data), ' files')
    # print("expected ", num_images)

    return att_info, names, images


def get_all_names(file):
    names = []
    with open(file, 'r') as file:
        lines = file.readlines()
        for l in lines:
            line = l.split()
            name = line[0][:-9]
            names.append(name)

    return names


def get_class_by_identity_name(file):
    idxs = {}
    with open(file, 'r') as file:
        lines = file.readlines()
        for l in lines:
            line = l.split()
            name = str(line[0][:-9])
            idxs[name] = int(line[1])

    return idxs


def norm(X, min, max):
    return (X - min) / (max - min)


def zscore_normalization(X):
    """
    Compute the z-score over image features X
    :param X: image embedding matrix, each row is an instance
    :return: z-score
    """

    z_score = stats.zscore(X, axis=1)

    return z_score

##########################################################################
# Preprocessing
##########################################################################

# Required files
lfw_identity_file = 'lfw_identity.txt'
lfw_attributes_files = '/home/cristianopatricio/Documents/Datasets/LFW/lfw_attributes.txt'

# Collect attributes
att_info, names, images = load_lfw_attrs(lfw_attributes_files)

all_names = get_all_names(lfw_identity_file)

delete_names = set(all_names).difference(set(names))
delete_names = list(delete_names)


getid = get_class_by_identity_name(lfw_identity_file)
ids_excluded = []
for name in delete_names:
    ids_excluded.append(getid[name])

# print(ids_excluded)

all_ids = [i for i in range(0, 5749)]
selected_ids = set(all_ids).difference(set(ids_excluded))
selected_ids = list(selected_ids)

# att
identity_info, image_info, images2 = load_lfw_identities(lfw_identity_file, images)

n_identities = len(selected_ids)
shuffle_identities = np.random.permutation(selected_ids)

# Split
training_classes = shuffle_identities[:math.ceil(n_identities * 0.6)]
validation_classes = shuffle_identities[math.ceil(n_identities * 0.6):math.ceil(n_identities * 0.8)]
test_classes = shuffle_identities[math.ceil(n_identities * 0.8):]
test_seen_classes = set(shuffle_identities).difference(test_classes)

print(len(training_classes))
print(len(validation_classes))
print(len(test_classes))

training_samples = []
test_seen_samples = []
for i in training_classes:
    samples = identity_info[i]
    if len(samples) < 2:
        training_samples.extend(samples)
    else:
        samples_train = identity_info[i][:-1]
        samples_test_seen = identity_info[i][-1:]
        training_samples.extend(samples_train)
        test_seen_samples.extend(samples_test_seen)

print(np.asarray(training_samples).shape)

val_samples = []
for i in validation_classes:
    samples = identity_info[i]
    if len(samples) < 2:
        val_samples.extend(samples)
    else:
        samples_train = identity_info[i][:-1]
        samples_test_seen = identity_info[i][-1:]
        val_samples.extend(samples_train)
        test_seen_samples.extend(samples_test_seen)

print(np.asarray(val_samples).shape)

test_samples = []
for i in test_classes:
    samples = identity_info[i]
    test_samples.extend(samples)

print(np.asarray(test_samples).shape)
print(np.asarray(test_seen_samples).shape)

print(str(int(len(training_samples) + len(val_samples) + len(test_samples) + len(test_seen_samples))))

# train_loc
train_loc = get_index_from_samples_v2(training_samples)

# val_loc
val_loc = get_index_from_samples_v2(val_samples)

# trainval_loc
trainval_loc = list(train_loc) + list(val_loc)

# test_unseen_loc
test_unseen_loc = get_index_from_samples_v2(test_samples)

# test_seen_loc
test_seen_loc = get_index_from_samples_v2(test_seen_samples)

# Convert into numpy array
train_loc = np.asarray(train_loc)
val_loc = np.asarray(val_loc)
trainval_loc = np.asarray(trainval_loc)
test_unseen_loc = np.asarray(test_unseen_loc)
test_seen_loc = np.asarray(test_seen_loc)

att_per_id = dict()
people_id_list = []
for image_id in image_info:
    person_id = image_info[image_id]
    people_id_list.append(person_id)

    if person_id not in att_per_id:
        att_per_id[person_id] = []

    att_per_id[person_id].append(att_info[image_id])

final_attrs = []
for i in people_id_list:
    min = -27.5264209415
    max = 14.4668098119
    attrs = att_per_id[i]
    attrs = np.asarray(attrs)
    attrs_norm = norm(attrs, min, max)
    #attrs_norm = zscore_normalization(attrs)
    final_attrs.append(np.mean(attrs_norm, axis=0))

# att
final_attrs = np.asarray(final_attrs)

dict = {
    'att': final_attrs,
    'train_loc': train_loc,
    'trainval_loc': trainval_loc,
    'val_loc': val_loc,
    'test_unseen_loc': test_unseen_loc,
    'test_seen_loc': test_seen_loc
}

print("Statistics")
print("att shape: ", final_attrs.shape)
print("train_loc shape: ", train_loc.shape)
print("trainval_loc shape: ", trainval_loc.shape)
print("val_loc shape: ", val_loc.shape)
print("test_unseen_loc shape: ", test_unseen_loc.shape)
print("test_seen_loc shape: ", test_seen_loc.shape)

# Save dict into a file
with open("lfw_att_split_orig_att.pickle", "wb") as f:
    pickle.dump(dict, f)