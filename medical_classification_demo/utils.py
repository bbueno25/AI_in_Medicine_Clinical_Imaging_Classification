"""
DOCSTRING
"""
import matplotlib.pyplot
import numpy
import os
import pandas
import PIL
import seaborn
import skimage
import time

class EDA:
    """
    DOCSTRING
    """
    def __init__(self):
        dict_labels = {
            0: "No DR",
            1: "Mild",
            2: "Moderate",
            3: "Severe",
            4: "Proliferative DR"}

    def __call__(self):
        labels = pandas.read_csv("labels/trainLabels.csv")
        plot_classification_frequency(labels, "level", "Retinopathy_vs_Frequency_All")
        plot_classification_frequency(labels, "level", "Retinopathy_vs_Frequency_Binary", True)

    def change_labels(df, category):
        """
        Changes the labels for a binary classification.
        Either the person has a degree of retinopathy, or they don't.

        INPUT
            df: Pandas DataFrame of the image name and labels
            category: column of the labels

        OUTPUT
            Column containing a binary classification of 0 or 1
        """
        return [1 if l > 0 else 0 for l in df[category]]

    def plot_classification_frequency(df, category, file_name, convert_labels=False):
        """
        Plots the frequency at which labels occur.

        INPUT
            df: Pandas DataFrame of the image name and labels
            category: category of labels, from 0 to 4
            file_name: file name of the image
            convert_labels: argument specified for converting to binary classification

        RETURN
            None
        """
        if convert_labels == True:
            labels['level'] = change_labels(labels, 'level')
        seaborn.set(style="whitegrid", color_codes=True)
        seaborn.countplot(x=category, data=labels)
        pyplot.title('Retinopathy vs Frequency')
        pyplot.savefig(file_name)
        return

class ImageToArray:
    """
    DOCSTRING
    """
    def __call__(self):
        start_time = time.time()
        labels = pandas.read_csv("../labels/trainLabels_master_256_v2.csv")
        print("Writing Train Array")
        X_train = convert_images_to_arrays_train('../data/train-resized-256/', labels)
        print(X_train.shape)
        print("Saving Train Array")
        save_to_array('../data/X_train.npy', X_train)
        print("--- %s seconds ---" % (time.time() - start_time))

    def change_image_name(self, df, column):
        """
        Appends the suffix '.jpeg' for all image names in the DataFrame

        INPUT
            df: Pandas DataFrame, including columns to be altered.
            column: The column that will be changed. Takes a string input.

        OUTPUT
            Pandas DataFrame, with a single column changed to include the
            aforementioned suffix.
        """
        return [i + '.jpeg' for i in df[column]]

    def convert_images_to_arrays_train(self, file_path, df):
        """
        Converts each image to an array, and appends each array to a new NumPy
        array, based on the image column equaling the image file name.

        INPUT
            file_path: Specified file path for resized test and train images.
            df: Pandas DataFrame being used to assist file imports.

        OUTPUT
            NumPy array of image arrays.
        """
        lst_imgs = [l for l in df['train_image_name']]
        return numpy.array([numpy.array(PIL.Image.open(file_path + img)) for img in lst_imgs])

    def save_to_array(self, arr_name, arr_object):
        """
        Saves data object as a NumPy file. Used for saving train and test arrays.

        INPUT
            arr_name: The name of the file you want to save.
                This input takes a directory string.
            arr_object: NumPy array of arrays. This object is saved as a NumPy file.

        OUTPUT
            NumPy array of image arrays
        """
        return numpy.save(arr_name, arr_object)

class PreprocessImages:
    """
    DOCSTRING
    """
    def __call__(self):
        start_time = time.time()
        trainLabels = pandas.read_csv('../labels/trainLabels.csv')
        trainLabels['image'] = [i + '.jpeg' for i in trainLabels['image']]
        trainLabels['black'] = numpy.nan
        trainLabels['black'] = find_black_images('../data/train-resized-256/', trainLabels)
        trainLabels = trainLabels.loc[trainLabels['black'] == 0]
        trainLabels.to_csv('trainLabels_master.csv', index=False, header=True)
        print("Completed")
        print("--- %s seconds ---" % (time.time() - start_time))

    def find_black_images(self, file_path, df):
        """
        Creates a column of images that are not black (numpy.mean(img) != 0)

        INPUT
            file_path: file_path to the images to be analyzed.
            df: Pandas DataFrame that includes all labeled image names.
            column: column in DataFrame query is evaluated against.

        OUTPUT
            Column indicating if the photo is pitch black or not.
        """
        lst_imgs = [l for l in df['image']]
        return [1 if numpy.mean(numpy.array(
            PIL.Image.open(file_path + img))) == 0 else 0 for img in lst_imgs]
    
    def rename_images(self, src_dir, new_prefix):
        """
        DOCSTRING
        """
        for file_name in os.listdir(src_dir):
            os.rename(
                os.path.join(src_dir, file_name),
                os.path.join(src_dir, new_prefix + file_name))
            print(file_name + ' -> ' + new_prefix + file_name)

class ReconcileLabels:
    """
    DOCSTRING
    """
    def __call__(self):
        trainLabels = pandas.read_csv("../labels/trainLabels_master.csv")
        lst_imgs = get_lst_images('../data/train-resized-256/')
        new_trainLabels = pandas.DataFrame({'image': lst_imgs})
        new_trainLabels['image2'] = new_trainLabels.image
        # remove the suffix from the image names
        new_trainLabels['image2'] = \
            new_trainLabels.loc[:, 'image2'].apply(lambda x: '_'.join(x.split('_')[0:2]))
        # strip and add jpeg back into file name
        new_trainLabels['image2'] = new_trainLabels.loc[:, 'image2'].apply(
            lambda x: '_'.join(x.split('_')[0:2]).strip('.jpeg') + '.jpeg')
        #trainLabels = trainLabels[0:10]
        new_trainLabels.columns = ['train_image_name', 'image']
        trainLabels = pandas.merge(trainLabels, new_trainLabels, how='outer', on='image')
        trainLabels.drop(['black'], axis=1, inplace=True)
        #print(trainLabels.head(100))
        trainLabels = trainLabels.dropna()
        print(trainLabels.shape)
        print("Writing CSV")
        trainLabels.to_csv('../labels/trainLabels_master_256_v2.csv', index=False, header=True)

    def get_lst_images(self, file_path):
        """
        Reads in all files from file path into a list.

        INPUT
            file_path: specified file path containing the images.

        OUTPUT
            List of image strings
        """
        return [i for i in os.listdir(file_path) if i != '.DS_Store']

class ResizeImages:
    """
    DOCSTRING
    """
    PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

    def __call__(self):
        self.crop_and_resize_images(
            path='../data/train/',
            new_path='../data/train-resized-256/',
            cropx=1800, cropy=1800, img_size=256)
        self.crop_and_resize_images(
            path='../data/test/',
            new_path='../data/test-resized-256/',
            cropx=1800, cropy=1800, img_size=256)

    def create_directory(self, directory):
        """
        Creates a new folder in the specified directory if the folder doesn't exist.

        INPUT
            directory: Folder to be created, called as "folder/".

        OUTPUT
            None
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        return

    def crop_and_resize_images(self, path, new_path, cropx, cropy, img_size=256):
        """
        Crops, resizes, and stores all images from a directory in a new directory.

        INPUT
            path: Path where the current, unscaled images are contained.
            new_path: Path to save the resized images.
            img_size: New size for the rescaled images.

        OUTPUT
            None
        """
        create_directory(new_path)
        dirs = [l for l in os.listdir(path) if l != '.DS_Store']
        total = 0
        for item in dirs:
            img = skimage.io.imread(path+item)
            y,x,channel = img.shape
            startx = x//2-(cropx//2)
            starty = y//2-(cropy//2)
            img = img[starty:starty+cropy,startx:startx+cropx]
            img = skimage.transform.resize(img, (256,256))
            skimage.io.imsave(str(new_path + item), img)
            total += 1
            print("Saving: ", item, total)
        return

class RotateImages:
    """
    DOCSTRING
    """
    #import cv2

    def __call__(self):
        start_time = time.time()
        trainLabels = pandas.read_csv("../labels/trainLabels_master.csv")
        trainLabels['image'] = trainLabels['image'].str.rstrip('.jpeg')
        trainLabels_no_DR = trainLabels[trainLabels['level'] == 0]
        trainLabels_DR = trainLabels[trainLabels['level'] >= 1]
        lst_imgs_no_DR = [i for i in trainLabels_no_DR['image']]
        lst_imgs_DR = [i for i in trainLabels_DR['image']]
        #lst_sample = [i for i in os.listdir('../data/sample/') if i != '.DS_Store']
        #lst_sample = [str(l.strip('.jpeg')) for l in lst_sample]
        # mirror Images with no DR one time
        print("Mirroring Non-DR Images")
        mirror_images('../data/train-resized-256/', 1, lst_imgs_no_DR)
        # rotate all images that have any level of DR
        print("Rotating 90 Degrees")
        rotate_images('../data/train-resized-256/', 90, lst_imgs_DR)
        print("Rotating 120 Degrees")
        rotate_images('../data/train-resized-256/', 120, lst_imgs_DR)
        print("Rotating 180 Degrees")
        rotate_images('../data/train-resized-256/', 180, lst_imgs_DR)
        print("Rotating 270 Degrees")
        rotate_images('../data/train-resized-256/', 270, lst_imgs_DR)
        print("Mirroring DR Images")
        mirror_images('../data/train-resized-256/', 0, lst_imgs_DR)
        print("Completed")
        print("--- %s seconds ---" % (time.time() - start_time))

    def rotate_images(self, file_path, degrees_of_rotation, lst_imgs):
        """
        Rotates image based on a specified amount of degrees

        INPUT
            file_path: file path to the folder containing images.
            degrees_of_rotation: Integer, specifying degrees to rotate the
            image. Set number from 1 to 360.
            lst_imgs: list of image strings.

        OUTPUT
            None
        """
        for l in lst_imgs:
            img = skimage.io.imread(file_path + str(l) + '.jpeg')
            img = skimage.transform.rotate(img, degrees_of_rotation)
            skimage.io.imsave(file_path + str(l) + '_' + str(degrees_of_rotation) + '.jpeg', img)
        return

    def mirror_images(self, file_path, mirror_direction, lst_imgs):
        """
        Mirrors image left or right, based on criteria specified.

        INPUT
            file_path: file path to the folder containing images.
            mirror_direction: criteria for mirroring left or right.
            lst_imgs: list of image strings.

        OUTPUT
            None
        """
        for l in lst_imgs:
            img = self.cv2.imread(file_path + str(l) + '.jpeg')
            img = self.cv2.flip(img, 1)
            self.cv2.imwrite(file_path + str(l) + '_mir' + '.jpeg', img)
        return

if __name__ == '__main__':
    preprocess_images = PreprocessImages()
    preprocess_images.rename_images('images/readme', 'readme-')
