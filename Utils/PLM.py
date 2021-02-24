import tensorflow as tf
import numpy as np
import pickle
from sklearn import preprocessing
import matplotlib.pyplot as plt

class PLM():

    def __init__(self):

        # Read data
        with open("vgg16.pickle", "rb") as f:
            dataset = pickle.load(f)

        with open("att_split_norm.pickle", "rb") as f:
            att_split = pickle.load(f)

        features = dataset['features']
        labels = dataset['labels']

        train_loc = att_split['train_loc']
        trainval_loc = att_split['trainval_loc']
        val_loc = att_split['val_loc']
        test_unseen_loc = att_split['test_unseen_loc']
        test_seen_loc = att_split['test_seen_loc']
        attribute = att_split['att']

        self.x_train = features[train_loc]
        self.y_train = labels[train_loc].astype(int)
        self.s_train = attribute[self.y_train]
        print("Shape x_train", self.x_train.shape)
        print("Shape y_train", self.y_train.shape)
        print("Shape s_train", self.s_train.shape)

        self.x_trainval = features[trainval_loc]
        self.y_trainval = labels[trainval_loc].astype(int)
        self.s_trainval = attribute[self.y_trainval]
        print("Shape x_trainval", self.x_trainval.shape)
        print("Shape y_trainval", self.y_trainval.shape)
        print("Shape s_trainval", self.s_trainval.shape)

        self.x_val = features[val_loc]
        self.y_val = labels[val_loc].astype(int)
        self.s_val = attribute[self.y_val]
        print("Shape x_val", self.x_val.shape)
        print("Shape y_val", self.y_val.shape)
        print("Shape s_val", self.s_val.shape)

        self.x_test_seen = features[test_seen_loc]
        self.y_test_seen = labels[test_seen_loc].astype(int)
        self.s_test_seen = attribute[self.y_test_seen]
        print("Shape x_test_seen", self.x_test_seen.shape)
        print("Shape y_test_seen", self.y_test_seen.shape)
        print("Shape s_test_seen", self.s_test_seen.shape)

        self.x_test_unseen = features[test_unseen_loc]
        self.y_test_unseen = labels[test_unseen_loc].astype(int)
        self.s_test_unseen = attribute[self.y_test_unseen.squeeze()]
        print("Shape x_test_unseen", self.x_test_unseen.shape)
        print("Shape y_test_unseen", self.y_test_unseen.shape)
        print("Shape s_test_unseen", self.s_test_unseen.shape)

        # GZSL
        self.x_test_gzsl = np.concatenate((self.x_test_seen, self.x_test_unseen), axis=0)
        self.s_test_gzsl = np.concatenate((self.s_test_seen, self.s_test_unseen), axis=0)
        self.y_test_gzsl = np.concatenate((self.y_test_seen, self.y_test_unseen), axis=0)
        print("Shape x_test_gzsl", self.x_test_gzsl.shape)
        print("Shape y_test_gzsl", self.y_test_gzsl.shape)
        print("Shape s_test_gzsl", self.s_test_gzsl.shape)

        self.decoded_y_seen = self.y_test_seen.copy()
        self.decoded_y_unseen = self.y_test_unseen.copy()
        self.decoded_y_test = self.y_test_gzsl.copy()

        self.train_labels_seen = np.unique(self.y_train)
        self.val_labels_unseen = np.unique(self.y_val)
        self.trainval_labels_seen = np.unique(self.y_trainval)
        self.test_labels_unseen = np.unique(self.y_test_unseen)
        self.test_labels_gzsl = np.unique(self.y_test_gzsl)

        # Normalize
        self.x_trainval = preprocessing.normalize(self.x_trainval, norm="l2")
        self.x_test_unseen = preprocessing.normalize(self.x_test_unseen, norm="l2")
        self.x_train = preprocessing.normalize(self.x_train, norm="l2")
        self.x_val = preprocessing.normalize(self.x_val, norm="l2")
        self.x_test_gzsl = preprocessing.normalize(self.x_test_gzsl, norm="l2")
        self.s_trainval = preprocessing.normalize(self.s_trainval, norm="l2")
        self.s_test_unseen = preprocessing.normalize(self.s_test_unseen, norm="l2")
        self.s_train = preprocessing.normalize(self.s_train, norm="l2")
        self.s_val = preprocessing.normalize(self.s_val, norm="l2")
        self.s_test_gzsl = preprocessing.normalize(self.s_test_gzsl, norm="l2")

        # Labels Encoder
        lbl = preprocessing.LabelEncoder()
        self.y_trainval = lbl.fit_transform(self.y_trainval)
        self.y_test_unseen = lbl.fit_transform(self.y_test_unseen)
        self.y_train = lbl.fit_transform(self.y_train)
        self.y_val = lbl.fit_transform(self.y_val)
        self.y_test_gzsl = lbl.fit_transform(self.y_test_gzsl)

    def cosine(self, v1, v2):
        return (np.dot(v1, v2))/(np.linalg.norm(v1) * np.linalg.norm(v2))

    def experiments(self):
        semantic_diffs = np.unique(self.s_test_unseen, axis=0)
        dict_semantic = {}
        get_idx_attr = {}
        for idx, sem in enumerate(semantic_diffs):
            dict_semantic[idx] = []
            for id, x in enumerate(self.s_test_unseen):
                if (x == sem).all():
                    dict_semantic[idx].append(id)
                    get_idx_attr[id] = idx


        return dict_semantic, get_idx_attr

    def my_loss_fn(self, y_true, y_pred):
        part_1 = np.exp(32 * (np.arccos(y_true.dot(y_pred) / (np.linalg.norm(y_true) * np.linalg.norm(y_pred)))))
        part_2 = np.sum(np.exp(32 * (np.arccos(y_true.dot(y_pred) / (np.linalg.norm(y_true) * np.linalg.norm(y_pred))))))

        return -np.sum(np.log(part_1 / part_2))

    def training(self):

        # Define Model
        inputs = tf.keras.Input(shape=(None, 40))
        x = tf.keras.layers.Dense(256, activation=tf.nn.relu)(inputs)
        outputs = tf.keras.layers.Dense(512, activation=tf.nn.relu)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Compile Model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=['accuracy'])

        # Fit Model
        history = model.fit(x=self.s_train,
                  y=self.x_train,
                  validation_data=(self.s_val, self.x_val),
                  shuffle=True,
                  epochs=100,
                  batch_size=32)

        return model, history

    def plot_metrics(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(len(acc))

        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

    def test(self, model, dict, get_idx_attr):

        #print("Shape s_test_unseen[0]: {0}".format(np.expand_dims(self.s_trainval[0], axis=0).shape))

        #search = np.expand_dims(self.s_trainval[0], axis=0)

        preds = model.predict(self.s_test_unseen)
        print("Predictions shape: {0}".format(preds.shape))

        distances = self.cosine(preds, self.x_test_unseen.T)
        print("Distances shape: {0}".format(distances.shape))

        distances_sort = np.argsort(distances, axis=1)
        top_5 = []
        for d in distances_sort:
            top_5.append(d[::-1][:5])
        top_5 = np.asarray(top_5)
        print("Top5 shape: {0}".format(top_5.shape))

        # Check Acc
        acc = {}
        for id, pred in enumerate(top_5):
            # Check if predicted image is in the images of the correspondent attribute
            if get_idx_attr[id] not in acc:
                acc[get_idx_attr[id]] = []
            count = 0
            for p in pred:
                if p in dict[get_idx_attr[id]]:
                    count += 1
            acc[get_idx_attr[id]].append(count / len(pred))

        acc_per_class = []
        for c in acc.values():
            acc_per_class.append(np.mean(c))

        acc_per_class = np.asarray(acc_per_class)
        print("Acc perclass: {0}".format(acc_per_class.shape))

        final_acc = np.mean(acc_per_class)
        print("Final acc shape: {0}".format(final_acc.shape))
        print("Accuracy: {:.3f}".format(final_acc))

        """
        gt = self.x_test_unseen[0]

        distances = self.cosine(preds, self.x_trainval.T)
        print("Cosine distances shape: {0}".format(distances.shape))

        top_5 = np.argsort(distances)[0, ::-1][:5]
        print(top_5)

        print("-----")
        print(self.x_trainval[0])
        print(self.x_trainval[top_5[0]])
        print("-----")

        star_id = 0
        for id, sem in enumerate(np.unique(self.s_test_unseen, axis=0)):
            if (sem == search).all():
                star_id = id

        print(dict[star_id])

        for i in top_5:
            if i in dict[star_id]:
                print("HOLA")
        """

        return acc

if __name__ == '__main__':
    plm = PLM()
    #model, history = plm.training()
    #plm.plot_metrics(history)
    #model.save("model.h5")
    dict, get_idx_attr = plm.experiments()
    #print(dict[0])
    #print(dict[0].shape)
    #model = plm.training()
    #print(plm.test(model))
    model = tf.keras.models.load_model('model.h5')
    plm.test(model, dict, get_idx_attr)