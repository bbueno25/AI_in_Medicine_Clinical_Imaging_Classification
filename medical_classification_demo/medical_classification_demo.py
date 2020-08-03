"""
DOCSTRING
"""
import numpy
import os
import pandas
import keras.callbacks
import keras.layers
import keras.models
import keras.utils
import sklearn.metrics
import sklearn.model_selection
import sklearn.utils
import skll.metrics

class CNN:
    """
    DOCSTRING
    """
    numpy.random.seed(1337)

    def __init__(self):
        batch_size = 512
        nb_classes = 2
        nb_epoch = 30
        img_rows, img_cols = 256, 256
        channels = 3
        nb_filters = 32
        kernel_size = (8, 8)

    def __call__(self):
        labels = pandas.read_csv("../labels/trainLabels_master_256_v2.csv")
        X = numpy.load("../data/X_train_256_v2.npy")
        y = numpy.array([1 if l >= 1 else 0 for l in labels['level']])
        #y = numpy.array(labels['level'])
        print("Splitting data into test/ train datasets")
        X_train, X_test, y_train, y_test = split_data(X, y, 0.2)
        print("Reshaping Data")
        X_train = reshape_data(X_train, img_rows, img_cols, channels)
        X_test = reshape_data(X_test, img_rows, img_cols, channels)
        print("X_train Shape: ", X_train.shape)
        print("X_test Shape: ", X_test.shape)
        input_shape = (img_rows, img_cols, channels)
        print("Normalizing Data")
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255
        y_train = np_utils.to_categorical(y_train, nb_classes)
        y_test = keras.utils.np_utils.to_categorical(y_test, nb_classes)
        print("y_train Shape: ", y_train.shape)
        print("y_test Shape: ", y_test.shape)
        print("Training Model")
        model = cnn_model(
            X_train, y_train, kernel_size, nb_filters,
            channels, nb_epoch, batch_size, nb_classes, nb_gpus=8)
        print("Predicting")
        y_pred = model.predict(X_test)
        score = model.evaluate(X_test, y_test, verbose=0)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        y_test = numpy.argmax(y_test, axis=1)
        y_pred = numpy.argmax(y_pred, axis=1)
        precision = sklearn.metrics.precision_score(y_test, y_pred)
        recall = sklearn.metrics.recall_score(y_test, y_pred)
        print("Precision: ", precision)
        print("Recall: ", recall)
        save_model(model=model, score=recall, model_name="DR_Two_Classes")
        print("Completed")

    def cnn_model(self, X_train, y_train, kernel_size, nb_filters, channels, nb_epoch, batch_size, nb_classes, nb_gpus):
        """
        Define and run the Convolutional Neural Network

        INPUT
            X_train: Array of NumPy arrays
            X_test: Array of NumPy arrays
            y_train: Array of labels
            y_test: Array of labels
            kernel_size: Initial size of kernel
            nb_filters: Initial number of filters
            channels: Specify if the image is grayscale (1) or RGB (3)
            nb_epoch: Number of epochs
            batch_size: Batch size for the model
            nb_classes: Number of classes for classification

        OUTPUT
            Fitted CNN model
        """
        model = keras.models.Sequential()
        model.add(keras.layers.convolutional.Conv2D(
            nb_filters, (kernel_size[0], kernel_size[1]),
            padding='valid', strides=1,
            input_shape=(img_rows, img_cols, channels), activation="relu"))
        model.add(keras.layers.convolutional.Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), activation="relu"))
        model.add(keras.layers.convolutional.Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), activation="relu"))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Flatten())
        print("Model flattened out to: ", model.output_shape)
        model.add(keras.layers.Dense(128))
        model.add(keras.layers.Activation('sigmoid'))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Dense(nb_classes))
        model.add(keras.layers.Activation('softmax'))
        model = keras.utils.multi_gpu_model(model, gpus=nb_gpus)
        model.compile(
            loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        stop = keras.callbacks.EarlyStopping(
            monitor='val_acc', min_delta=0.001, patience=2, verbose=0, mode='auto')
        tensor_board = keras.callbacks.TensorBoard(
            log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
        model.fit(
            X_train, y_train, batch_size=batch_size, epochs=nb_epoch,
            verbose=1, validation_split=0.2, class_weight='auto', callbacks=[stop, tensor_board])
        return model

    def reshape_data(self, arr, img_rows, img_cols, channels):
        """
        Reshapes the data into format for CNN.

        INPUT
            arr: Array of NumPy arrays.
            img_rows: Image height
            img_cols: Image width
            channels: Specify if the image is grayscale (1) or RGB (3)

        OUTPUT
            Reshaped array of NumPy arrays.
        """
        return arr.reshape(arr.shape[0], img_rows, img_cols, channels)

    def save_model(self, model, score, model_name):
        """
        Saves Keras model to an h5 file, based on precision_score

        INPUT
            model: Keras model object to be saved
            score: Score to determine if model should be saved.
            model_name: name of model to be saved

        RETURN
            None
        """
        if score >= 0.75:
            print("Saving Model")
            model.save("../models/" + model_name + "_recall_" + str(round(score, 4)) + ".h5")
        else:
            print("Model Not Saved.  Score: ", score)
        return

    def split_data(self, X, y, test_data_size):
        """
        Split data into test and training datasets.

        INPUT
            X: NumPy array of arrays
            y: Pandas series, which are the labels for input array X
            test_data_size: size of test/train split. Value from 0 to 1

        OUPUT
            Four arrays: X_train, X_test, y_train, and y_test
        """
        return sklearn.model_selection.train_test_split(X, y, test_size=test_data_size, random_state=42)

class CNNMulti:
    """
    DOCSTRING
    """
    numpy.random.seed(1337)

    def __call__(self):
        # Specify GPU's to Use
        os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
        # Specify parameters before model is run.
        batch_size = 1000
        nb_classes = 5
        nb_epoch = 30
        img_rows, img_cols = 256, 256
        channels = 3
        nb_filters = 32
        kernel_size = (8,8)
        # Import data
        labels = pandas.read_csv("../labels/trainLabels_master_256_v2.csv")
        X = numpy.load("../data/X_train_256_v2.npy")
        y = numpy.array(labels['level'])
        # Class Weights (for imbalanced classes)
        print("Computing Class Weights")
        weights = sklearn.utils.class_weight.compute_class_weight('balanced', numpy.unique(y), y)
        print("Splitting data into test/ train datasets")
        X_train, X_test, y_train, y_test = split_data(X, y, 0.2)
        print("Reshaping Data")
        X_train = reshape_data(X_train, img_rows, img_cols, channels)
        X_test = reshape_data(X_test, img_rows, img_cols, channels)
        print("X_train Shape: ", X_train.shape)
        print("X_test Shape: ", X_test.shape)
        input_shape = (img_rows, img_cols, channels)
        print("Normalizing Data")
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255
        y_train = keras.utils.np_utils.to_categorical(y_train, nb_classes)
        y_test = keras.utils.np_utils.to_categorical(y_test, nb_classes)
        print("y_train Shape: ", y_train.shape)
        print("y_test Shape: ", y_test.shape)
        print("Training Model")
        model = cnn_model(
            X_train, X_test, y_train, y_test, kernel_size,
            nb_filters, channels, nb_epoch, batch_size, nb_classes)
        print("Predicting")
        y_pred = model.predict(X_test)
        score = model.evaluate(X_test, y_test, verbose=0)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        y_pred = [numpy.argmax(y) for y in y_pred]
        y_test = [numpy.argmax(y) for y in y_test]
        precision = sklearn.metrics.precision_score(y_test, y_pred, average='weighted')
        recall = sklearn.metrics.recall_score(y_test, y_pred, average='weighted')
        print("Precision: ", precision)
        print("Recall: ", recall)
        save_model(model=model, score=recall, model_name="DR_Two_Classes")
        print("Completed")

    def cnn_model(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        kernel_size,
        nb_filters,
        channels,
        nb_epoch,
        batch_size,
        nb_classes):
        """
        Define and run the Convolutional Neural Network.

        INPUT
            X_train: Array of NumPy arrays
            X_test: Array of NumPy arrays
            y_train: Array of labels
            y_test: Array of labels
            kernel_size: Initial size of kernel
            nb_filters: Initial number of filters
            channels: Specify if the image is grayscale (1) or RGB (3)
            nb_epoch: Number of epochs
            batch_size: Batch size for the model
            nb_classes: Number of classes for classification

        OUTPUT
            Fitted CNN model
        """
        model = keras.models.Sequential()
        model.add(keras.layers.convolutional.Conv2D(
            nb_filters, (kernel_size[0], kernel_size[1]),
            padding='valid', strides=4,
            input_shape=(img_rows, img_cols, channels)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.convolutional.Conv2D(nb_filters, (kernel_size[0], kernel_size[1])))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.convolutional.Conv2D(nb_filters, (kernel_size[0], kernel_size[1])))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
        kernel_size = (16,16)
        model.add(keras.layers.convolutional.Conv2D(64, (kernel_size[0], kernel_size[1])))
        model.add(keras.layers.Activation('relu'))
        #model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
        model.add(keras.layers.Flatten())
        print("Model flattened out to: ", model.output_shape)
        model.add(keras.layers.Dense(128))
        model.add(keras.layers.Activation('sigmoid'))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Dense(nb_classes))
        model.add(keras.layers.Activation('softmax'))
        model.compile(
            loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        stop = keras.callbacks.EarlyStopping(
            monitor='val_acc', min_delta=0.001, patience=2, verbose=0, mode='auto')
        tensor_board = keras.callbacks.TensorBoard(
            log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
        model.fit(
            X_train,y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1,
            validation_split=0.2, class_weight=weights, callbacks=[stop, tensor_board])
        return model

    def reshape_data(self, arr, img_rows, img_cols, channels):
        """
        Reshapes the data into format for CNN.

        INPUT
            arr: Array of NumPy arrays.
            img_rows: Image height
            img_cols: Image width
            channels: Specify if the image is grayscale (1) or RGB (3)

        OUTPUT
            Reshaped array of NumPy arrays.
        """
        return arr.reshape(arr.shape[0], img_rows, img_cols, channels)
    
    def save_model(self, model, score, model_name):
        """
        Saves Keras model to an h5 file, based on precision_score

        INPUT
            model: Keras model object to be saved
            score: Score to determine if model should be saved.
            model_name: name of model to be saved
        """
        if score >= 0.75:
            print("Saving Model")
            model.save("../models/" + model_name + "_recall_" + str(round(score,4)) + ".h5")
        else:
            print("Model Not Saved. Score: ", score)

    def split_data(self, X, y, test_data_size):
        """
        Split data into test and training datasets.

        INPUT
            X: NumPy array of arrays
            y: Pandas series, which are the labels for input array X
            test_data_size: size of test/train split. Value from 0 to 1

        OUPUT
            Four arrays: X_train, X_test, y_train, and y_test
        """
        return sklearn.model_selection.train_test_split(X, y, test_size=test_data_size, random_state=42)

class EyeNet:
    """
    DOCSTRING
    """
    numpy.random.seed(1337)

    def __init__(self):
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.test_data_size = None
        self.weights = None
        self.model = None
        self.nb_classes = None
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.n_gpus = 8

    def __call__(self):
        self.split_data(
            y_file_path="../labels/trainLabels_master_256_v2.csv",
            X="../data/X_train_256_v2.npy")
        self.reshape_data(img_rows=256, img_cols=256, channels=3, nb_classes=5)
        model = self.cnn_model(
            nb_filters=32, kernel_size=(4, 4), batch_size=512, nb_epoch=50)
        precision, recall, f1, cohen_kappa, quad_kappa  = self.predict()
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1: ", f1)
        print("Cohen Kappa Score", cohen_kappa)
        print("Quadratic Kappa: ", quad_kappa)
        self.save_model(score=recall, model_name="DR_Class")

    def cnn_model(self, nb_filters, kernel_size, batch_size, nb_epoch):
        """
        Define and run the Convolutional Neural Network.

        INPUT
            X_train: Array of NumPy arrays
            X_test: Array of NumPy arrays
            y_train: Array of labels
            y_test: Array of labels
            kernel_size: Initial size of kernel
            nb_filters: Initial number of filters
            channels: Specify if the image is grayscale (1) or RGB (3)
            nb_epoch: Number of epochs
            batch_size: Batch size for the model
            nb_classes: Number of classes for classification

        OUTPUT
            Fitted CNN model
        """
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.convolutional.Conv2D(
            nb_filters, (kernel_size[0], kernel_size[1]), padding="valid", strides=1,
            input_shape=(self.img_rows, self.img_cols, self.channels), activation="relu"))
        self.model.add(keras.layers.convolutional.Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), activation="relu"))
        self.model.add(keras.layers.convolutional.Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), activation="relu"))
        self.model.add(keras.layers.MaxPooling2D(pool_size=(8, 8)))
        self.model.add(keras.layers.Flatten())
        print("Model flattened out to: ", self.model.output_shape)
        self.model.add(keras.layers.Dense(2048, activation="relu"))
        self.model.add(keras.layers.Dropout(0.25))
        self.model.add(keras.layers.Dense(2048, activation="relu"))
        self.model.add(keras.layers.Dropout(0.25))
        self.model.add(keras.layers.Dense(self.nb_classes, activation="softmax"))
        self.model = keras.utils.multi_gpu_model(self.model, gpus=self.n_gpus)
        self.model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        stop = keras.callbacks.EarlyStopping(
            monitor="val_acc", min_delta=0.001, patience=2, mode="auto")
        self.model.fit(
            self.X_train, self.y_train, batch_size=batch_size, epochs=nb_epoch,
            verbose=1, validation_split=0.2, class_weight=self.weights, callbacks=[stop])
        return self.model

    def predict(self):
        """
        Predicts the model output, and computes precision, recall, and F1 score.

        INPUT
            None

        OUTPUT
            Precision, Recall, and F1 score
        """
        predictions = self.model.predict(self.X_test)
        predictions = numpy.argmax(predictions, axis=1)
        #predictions[predictions >=1] = 1 # Remove when non binary classifier
        self.y_test = numpy.argmax(self.y_test, axis=1)
        precision = sklearn.metrics.precision_score(self.y_test, predictions, average="micro")
        recall = sklearn.metrics.recall_score(self.y_test, predictions, average="micro")
        f1 = sklearn.metrics.f1_score(self.y_test, predictions, average="micro")
        cohen_kappa = sklearn.metrics.cohen_kappa_score(self.y_test, predictions)
        quad_kappa = skll.metrics.kappa(self.y_test, predictions, weights='quadratic')
        return precision, recall, f1, cohen_kappa, quad_kappa

    def reshape_data(self, img_rows, img_cols, channels, nb_classes):
        """
        Reshapes arrays into format for MXNet

        INPUT
            img_rows: Array (image) height
            img_cols: Array (image) width
            channels: Specify if image is grayscale(1) or RGB (3)
            nb_classes: number of image classes/ categories

        OUTPUT
            None
        """
        self.nb_classes = nb_classes
        self.X_train = self.X_train.reshape(self.X_train.shape[0], img_rows, img_cols, channels)
        self.X_train = self.X_train.astype("float32")
        self.X_train /= 255
        self.y_train = keras.utils.np_utils.to_categorical(self.y_train, self.nb_classes)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], img_rows, img_cols, channels)
        self.X_test = self.X_test.astype("float32")
        self.X_test /= 255
        self.y_test = keras.utils.np_utils.to_categorical(self.y_test, self.nb_classes)
        print("X_train Shape:", self.X_train.shape)
        print("X_test Shape:", self.X_test.shape)
        print("y_train Shape:", self.y_train.shape)
        print("y_test Shape:", self.y_test.shape)
        return

    def save_model(self, score, model_name):
        """
        Saves the model, based on scoring criteria input.

        INPUT
            score: Scoring metric used to save model or not.
            model_name: name for the model to be saved.

        OUTPUT
            None
        """
        if score >= 0.75:
            print("Saving Model")
            self.model.save("../models/" + model_name + "_recall_" + str(round(score, 4)) + ".h5")
        else:
            print("Model Not Saved. Score: ", score)
        return

    def split_data(self, y_file_path, X, test_data_size=0.2):
        """
        Split data into test and training data sets.

        INPUT
            y_file_path: path to CSV containing labels
            X: NumPy array of arrays
            test_data_size: size of test/train split. Value from 0 to 1

        OUTPUT
            None
        """
        #labels = pandas.read_csv(y_file_path, nrows=60)
        labels = pandas.read_csv(y_file_path)
        self.X = numpy.load(X)
        self.y = numpy.array(labels['level'])
        self.weights = sklearn.utils.class_weight.compute_class_weight(
            'balanced', numpy.unique(self.y), self.y)
        self.test_data_size = test_data_size
        self.X_train, self.X_test, self.y_train, self.y_test = \
            sklearn.model_selection.train_test_split(
                self.X, self.y, test_size=self.test_data_size, random_state=42)
        return
