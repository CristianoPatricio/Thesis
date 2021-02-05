import numpy as np
import math
import re
import pickle

#################################################################
# Auxiliary Functions
#################################################################

pubfig_attributes_names = ["Male", "Asian", "White", "Black", "Baby", "Child", "Youth", "Middle Aged", "Senior",
                           "Black Hair", "Blond Hair", "Brown Hair", "Bald", "No Eyewear", "Eyeglasses", "Sunglasses",
                           "Mustache", "Smiling", "Frowning", "Chubby", "Blurry", "Harsh Lighting", "Flash",
                           "Soft Lighting", "Outdoor", "Curly Hair", "Wavy Hair", "Straight Hair", "Receding Hairline",
                           "Bangs", "Sideburns", "Fully Visible Forehead", "Partially Visible Forehead",
                           "Obstructed Forehead", "Bushy Eyebrows", "Arched Eyebrows", "Narrow Eyes", "Eyes Open",
                           "Big Nose", "Pointy Nose", "Big Lips", "Mouth Closed", "Mouth Slightly Open",
                           "Mouth Wide Open", "Teeth Not Visible", "No Beard", "Goatee", "Round Jaw", "Double Chin",
                           "Wearing Hat", "Oval Face", "Square Face", "Round Face", "Color Photo", "Posed Photo",
                           "Attractive Man", "Attractive Woman", "Indian", "Gray Hair", "Bags Under Eyes",
                           "Heavy Makeup", "Rosy Cheeks", "Shiny Skin", "Pale Skin", "5 o' Clock Shadow",
                           "Strong Nose-Mouth Lines", "Wearing Lipstick", "Flushed Face", "High Cheekbones",
                           "Brown Eyes", "Wearing Earrings", "Wearing Necktie", "Wearing Necklace"]


def image_name2image_id(image_name):
    return int(image_name[:-4])


def image2id_dict():
    dict = {}
    with open("pubfig_identity.txt") as file:
        lines = file.readlines()
        for idx, l in enumerate(lines):
            line = l.split()
            image_name = line[0]
            image_id = idx
            dict[image_name] = image_id

    return dict


def load_pubfig_identities(pubfig_identities_file):
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
    with open(pubfig_identities_file, "r") as identities:
        lines = identities.readlines()
        for idx, identity in enumerate(lines):
            line = re.split("\t", identity)
            image_name = line[0]
            image_id = image_name[:-4]
            identity_id = float(line[1])

            if identity_id not in identity_info:
                identity_info[identity_id] = []
            identity_info[identity_id].append(image_name)
            image_info[image_id] = identity_id

    return identity_info, image_info


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
    image2id = image2id_dict()
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


def load_pubfig_attrs(pubfig_atts_file):
    """"
    input: pubfig_atts_file - path to the file containing PubFig attributes

        pubfig_attributes.txt
        image_name att_1 att_2 att_3 ... att_11


    output: identity_info - dictionary of the bb names per image name

        att_info[image_id] -> (att_n_1, att_n_2, ..., att_n_40)

    """

    att_info = dict()
    with open(pubfig_atts_file, 'r') as atributes_file:
        lines = atributes_file.readlines()
        assert (len(lines) > 3)

        # second line is the header
        for idx, line in enumerate(lines[2:]):

            values = re.split("\t", line)

            pre_name = values[0].replace(" ", "_")
            id = int(values[1])

            if id < 10:
                name = pre_name + str("_") + str('000') + str(id)
            elif 10 <= id < 100:
                name = pre_name + str("_") + str('00') + str(id)
            elif 100 <= id < 1000:
                name = pre_name + str("_") + str('0') + str(id)
            else:
                name = pre_name + str("_") + str(id)

            image_id = name

            attibutes_arr = []
            for i in range(0, len(pubfig_attributes_names)):
                value = values[i + 2]
                attibutes_arr.append(float(value))

            att_info[image_id] = np.array(attibutes_arr)

    return att_info


def norm(X, min, max):
    return (X - min) / (max - min)


##########################################################################
# Preprocessing
##########################################################################

if __name__ == '__main__':

    pubfig_identity_file = 'pubfig_identity.txt'

    # att
    identity_info, image_info = load_pubfig_identities(pubfig_identity_file)

    most_populated_classes = {}
    for i in range(0, 200):
        most_populated_classes[i] = len(identity_info[i])

    sort_orders = sorted(most_populated_classes.items(), key=lambda x: x[1], reverse=True)

    selected_classes = []
    for i in sort_orders[:50]:  # choose a number until 200
        selected_classes.append(i[0])

    n_identities = 50  # choose a number until 10177
    shuffle_identities = np.random.permutation(selected_classes)

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
            samples_train = identity_info[i][:-20]
            samples_test_seen = identity_info[i][-20:]
            training_samples.extend(samples_train)
            test_seen_samples.extend(samples_test_seen)

    print(np.asarray(training_samples).shape)

    val_samples = []
    for i in validation_classes:
        samples = identity_info[i]
        if len(samples) < 2:
            val_samples.extend(samples)
        else:
            samples_train = identity_info[i][:-20]
            samples_test_seen = identity_info[i][-20:]
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

    # Collect attributes
    att_info = load_pubfig_attrs('pubfig_attributes.txt')

    att_per_id = dict()
    for image_id in image_info:

        person_id = image_info[image_id]

        if person_id not in att_per_id:
            att_per_id[person_id] = []

        att_per_id[person_id].append(att_info[image_id])

    final_attrs = []
    for i in range(0, 200):
        attrs = att_per_id[i]
        attrs = np.asarray(attrs)
        attrs_norm = norm(attrs, -33.958466016, 23.3353731125)
        final_attrs.append(np.abs(np.mean(attrs_norm, axis=0)))

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
    with open("pubfig_att_split_50_norm.pickle", "wb") as f:
        pickle.dump(dict, f)