import numpy as np
import math
import re
import pickle


#################################################################
# Auxiliary Functions
#################################################################

celeb_attributes_names =  ["5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young"]


def image_name2image_id(image_name):
    return int(image_name[:-4])


def load_celeba_identities(celeba_identities_files):
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
    with open(celeba_identities_files) as identities:
        lines = identities.readlines()
        for identity in lines:
            identity = identity.rstrip().lstrip().split()
            # we have 2 infos per line, image name and identity id
            if len(identity) != 2:
                continue
            image_name = identity[0]
            identity_id = int(identity[1])

            if identity_id not in identity_info:
                identity_info[identity_id] = []
            identity_info[identity_id].append(image_name)
            image_info[image_name2image_id(image_name)] = identity_id

    return identity_info, image_info


def get_index_from_samples(samples, celeba_identity_file):
    idxs = []
    with open(celeba_identity_file, "r") as identities:
        lines = identities.readlines()
        for idx, line in enumerate(lines):
            line = line.rstrip().split()
            if line[0] in samples:
                idxs.append(idx)

    idxs = np.asarray(idxs)

    return idxs


def get_index_from_samples_v2(samples):
    idxs = []
    for sample in samples:
        idxs.append(int(sample[:-4])-1)

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


def load_celeba_attrs(celeba_attributes_file):

    """"
    input: celeba_attributes_file - path to the file containing CELEB-A attributes

        list_attr_celeba.txt
        N (HEADER)
        attribute names (HEADER)
        image_id_1 att_1_1 att_1_2 ... att_1_40
        ...
        image_id_n att_n_1 att_n_2 ... att_n_40


    output: identity_info - dictionary of the bb names per image name

        att_info[image_id] -> (att_n_1, att_n_2, ..., att_n_40)

    """
    att_info = dict()
    with open(celeba_attributes_file, 'r') as atributes_file:
        lines = atributes_file.readlines()
        assert(len(lines) > 3)

        # first line is the number of images line
        num_images = int(lines[0])
        # second line is the header
        for line in lines[2:]:

            values = re.split(" +", line)
            image_id = image_name2image_id(values[0])

            attibutes_arr = []
            for i in range(0, len(celeb_attributes_names)):
                attr = celeb_attributes_names[i]
                value = values[i + 1]

                attibutes_arr.append(int(value))

            att_info[image_id] = np.array(attibutes_arr)

    # print("found ", len(celeb_data), ' files')
    # print("expected ", num_images)

    return att_info



##########################################################################
# Preprocessing
##########################################################################

celeba_identity_file = '/home/cristianopatricio/Documents/Datasets/Celeb-A/data/identity_CelebA.txt'

# att
identity_info, image_info = load_celeba_identities(celeba_identity_file)

n_identities = len(identity_info.keys())

most_populated_classes = {}
for i in range(1, 10178):
    most_populated_classes[i] = len(identity_info[i])

sort_orders = sorted(most_populated_classes.items(), key=lambda x: x[1], reverse=True)

selected_classes = []
for i in sort_orders[:50]: # choose a number until 10177
    selected_classes.append(i[0])

n_classes = 50  # choose a number until 10177
shuffle_identities = np.random.permutation(selected_classes)

# Split
training_classes = shuffle_identities[:math.ceil(n_classes * 0.6)]
validation_classes = shuffle_identities[math.ceil(n_classes * 0.6):math.ceil(n_classes * 0.8)]
test_classes = shuffle_identities[math.ceil(n_classes * 0.8):]
test_seen_classes = set(shuffle_identities).difference(test_classes)

print(len(training_classes))
print(len(validation_classes))
print(len(test_classes))

training_samples = []
test_seen_samples = []
for i in training_classes:
    samples = identity_info[i][:-2]
    samples_test_seen = identity_info[i][-2:]
    training_samples.extend(samples)
    test_seen_samples.extend(samples_test_seen)

print(np.asarray(training_samples).shape)

val_samples = []
for i in validation_classes:
    samples = identity_info[i][:-2]
    samples_test_seen = identity_info[i][-2:]
    val_samples.extend(samples)
    test_seen_samples.extend(samples_test_seen)

print(np.asarray(val_samples).shape)

test_samples = []
for i in test_classes:
    samples = identity_info[i]
    test_samples.extend(samples)

print(np.asarray(test_samples).shape)
print(np.asarray(test_seen_samples).shape)

print(str(int(len(training_samples)+len(val_samples)+len(test_samples)+len(test_seen_samples))))

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

# Collect attributes
att_info = load_celeba_attrs('/home/cristianopatricio/Documents/Datasets/Celeb-A/data/list_attr_celeba.txt')

att_per_id = dict()
for image_id in image_info:

    person_id = image_info[image_id]

    if person_id not in att_per_id:
        att_per_id[person_id] = []

    att_per_id[person_id].append(att_info[image_id])

final_attrs = []
for i in range(1, n_identities+1):
    attrs = att_per_id[i]
    attrs = np.asarray(attrs)
    final_attrs.append(np.abs(np.mean(attrs, axis=0)))

# att
final_attrs = np.asarray(final_attrs)

dict = {
    'att' : final_attrs,
    'train_loc' : train_loc,
    'trainval_loc' : trainval_loc,
    'val_loc' : val_loc,
    'test_unseen_loc' : test_unseen_loc,
    'test_seen_loc' : test_seen_loc
}

# Save dict into a file
with open("att_split_50.pickle", "wb") as f:
    pickle.dump(dict, f)
