# original source code from https://github.com/XifengGuo/CapsNet-Keras
#%matplotlib inline
import numpy as np
from keras.utils import to_categorical
from hyperparams import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import datasets

#from datasets.coil20.feed import feed
#path = "E:/DeepLearning/envs/myenvs/Lib/site-packages/keras/datasets"
#feed(feed_path=path, dataset_type='processed')

num_classes = NUM_CLASSES

def load_xray():
    # the data, shuffled and split between train and test sets
    #from keras.datasets import mnist

    data_dir = 'E:/DeepLearning/envs/myenvs/kobby/ImageToMNIST/xray/'
    (x_train, y_train), (x_test, y_test) = data_dir

    x_train = x_train.reshape(-1,28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)

def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1,28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)

'''
def load_coil_20():
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = coil20.load_data()

    x_train = x_train.reshape(-1,28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)
'''





def load_fashion_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.reshape(-1,28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)

def load_cifar10():
    # the data, shuffled and split between train and test sets
    from keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    return (x_train, y_train), (x_test, y_test)

def load_cifar100():
    # the data, shuffled and split between train and test sets
    from keras.datasets import cifar100
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    return (x_train, y_train), (x_test, y_test)




        #from tensorflow.keras.preprocessing.image import ImageDataGenerator
    IMG_SIZE = (48, 48)
    datagen = ImageDataGenerator(samplewise_center=True,
                              samplewise_std_normalization=True,
                              horizontal_flip = True,
                              vertical_flip = False,
                              height_shift_range= 0.05,
                              width_shift_range=0.1,
                              rotation_range=5,
                              shear_range = 0.1,
                              fill_mode = 'reflect',
                              zoom_range=0.15)

    train_generator = datagen.flow_from_directory(
            'D:/Dataset/chest-xray-pneumonia/chest_xray/train',
            target_size=IMG_SIZE,
            color_mode = 'grayscale',
            batch_size=32,
            class_mode='binary')

    val_generator = next(datagen.flow_from_directory(
            'D:/Dataset/chest-xray-pneumonia/chest_xray/val',
            target_size=IMG_SIZE,
            color_mode = 'grayscale',
            batch_size=32,
            class_mode='binary')) # one big batch

    test_generator = next(datagen.flow_from_directory(
            'D:/Dataset/chest-xray-pneumonia/chest_xray/test',
            target_size=IMG_SIZE,
            color_mode = 'grayscale',
            batch_size=180,
            class_mode='binary')) # one big batch


    ###### New Data Augumentation #######



#====================ADDED BY ME TO SAVE CSV=======================================
import numpy as np
from matplotlib import pyplot as plt
import csv
import math
import pandas
#%matplotlib inline
from sklearn.metrics import confusion_matrix
import itertools
#from Pneucaps_Model import base_model


def plot_log(filename, show=True):

    data = pandas.read_csv(filename)

    fig = plt.figure(figsize=(7,9))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    for key in data.keys():
        if key.find('loss') >= 0 and not key.find('val') >= 0:  # training loss
            plt.plot(data['epoch'].values, data[key].values, label=key)
    plt.legend()
    plt.grid(linestyle='--')
    plt.title('Training loss')
    plt.savefig(save_dir+'Training loss')

    fig.add_subplot(212)
    for key in data.keys():
        if key.find('acc') >= 0:  # acc
            plt.plot(data['epoch'].values, data[key].values, label=key)
    plt.legend()
    plt.grid(linestyle='--')
    plt.title('Training and validation accuracy')
    plt.savefig(save_dir+'Training and validation accuracy')

    # fig.savefig('result/log.png')
    if show:
        plt.show()


def combine_images(generated_images, height=None, width=None):
    num = generated_images.shape[0]
    if width is None and height is None:
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
    elif width is not None and height is None:  # height not given
        height = int(math.ceil(float(num)/width))
    elif height is not None and width is None:  # width not given
        width = int(math.ceil(float(num)/height))

    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image



if __name__=="__main__":
    from keras import callbacks
    save_dir = 'E:/DeepLearning/envs/myenvs/kobby/SCCapsule/models' ## Choose your prefered directory to save the train history CSV file
    
    plot_log(save_dir + '/BrainCancer(CapsNet(MLAF-CapsNet)).csv')
    #plt.
