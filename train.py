train#!/usr/bin/env python
# coding: utf-8

# ## Model training records
# * This code same with 'train.py'
# * Check the hyper parameters in 'hyperparams.py'

# In[ ]:

############ Proposed MLAF-CapsNet ###############

from Pneumonia_models.DSG_Caps import base_model

#################################################

####### Baseline Models #####################

#from Pneumonia_models.DSG_Caps import base_model


###############################################


from utils import *
from hyperparams import * # Check the hyper parameters in 'hyperparams.py'
from keras import callbacks, optimizers
#from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
import os
from keras.layers import Dense, Lambda
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.datasets import fashion_mnist
from datetime import datetime
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
#from sklearn.multiclass import oneVsRestClassifier
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
#from plt.sytle.use()
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix,precision_recall_curve, average_precision_score
import random
from itertools import count
from scipy import interp
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from keras.callbacks import EarlyStopping
import keras
import numpy as np
import tensorflow as tf
tf.set_random_seed(1)
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
#from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes


from keras.applications import vgg16, inception_v3, resnet50, mobilenet

    ### RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier




data =  'E:/DeepLearning/envs/myenvs/kobby/xray/train'
data1 = 'E:/DeepLearning/envs/myenvs/kobby/xray/test'

samples = len(data)
val_sample = len(data1)
BATCH_SIZE = 8

#steps_per_epoch = samples // batch

def margin_loss(y_true, y_pred):
    # original source code from
    # https://github.com/XifengGuo/CapsNet-Keras
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + 0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
    return K.mean(K.sum(L, 1))



def masked_maximum(data, mask, dim=1):
    """Computes the axis wise maximum over chosen elements.

    Args:
      data: 2-D float `Tensor` of size [n, m].
      mask: 2-D Boolean `Tensor` of size [n, m].
      dim: The dimension over which to compute the maximum.

    Returns:
      masked_maximums: N-D `Tensor`.
        The maximized dimension is of size 1 after the operation.
    """
    axis_minimums = math_ops.reduce_min(data, dim, keepdims=True)
    masked_maximums = math_ops.reduce_max(
        math_ops.multiply(data - axis_minimums, mask), dim,
        keepdims=True) + axis_minimums
    return masked_maximums

def masked_minimum(data, mask, dim=1):
    """Computes the axis wise minimum over chosen elements.

    Args:
      data: 2-D float `Tensor` of size [n, m].
      mask: 2-D Boolean `Tensor` of size [n, m].
      dim: The dimension over which to compute the minimum.

    Returns:
      masked_minimums: N-D `Tensor`.
        The minimized dimension is of size 1 after the operation.
    """
    axis_maximums = math_ops.reduce_max(data, dim, keepdims=True)
    masked_minimums = math_ops.reduce_min(
        math_ops.multiply(data - axis_maximums, mask), dim,
        keepdims=True) + axis_maximums
    return masked_minimums

def pose_loss(y_true, y_pred):
    """
    :param y_true: [None, n_classes, n_instance,pose]
    :param y_pred: [None, n_classes, n_instance,pose]
    :return: a scalar loss value.
    """
    loss = K.sum( K.square(y_true-y_pred),-1)

   
    return loss


if __name__ == "__main__":
    # Load dataset
    
    
    #(x_train, y_train), (x_test, y_test) = load_mnist()  #Train MNIST dataset
    #(x_train, y_train), (x_test, y_test) = load_coil_20()  #Train MNIST dataset
    #(x_train, y_train), (x_test, y_test) = load_fashion_mnist()  #Train Fashion_MNIST dataset
    (x_train, y_train), (x_test, y_test) = load_cifar10()#train cifar10
    #(x_train, y_train), (x_test, y_test) = load_cifar100()#train cifar100
    x_shape = x_train[0].shape
    y_shape = y_train[0].shape

    print('x_shape:===>', x_shape)

    
    train_model, eval_model = densenet(input_shape = x_shape, n_classes=NUM_CLASSES, f=32)
    #SHAPE_Y, SHAPE_X = 32, 32
    #InputA = (SHAPE_Y, SHAPE_X,3)
    #InputB = (SHAPE_Y, SHAPE_X,3)


    #train_model = base_model(inputs=[InputA, InputB], outputs=[combined])
    #out1 = train_model(InputA)
    #out2 = train_model(InputB)

    MODEL_PATH = "{}_weights.best.hdf5".format('BrainCancer(CapsNet(MLAF-CapsNet))')
    #MODEL_PATHN = "{}_weights.best.hdf5".format('Fashion_Mnist')
    #train_model.save("FashionMNIST_model.h5")

    # Init callback functions
    checkpoint = callbacks.ModelCheckpoint(MODEL_PATH,
                                           monitor='val_capsnet_acc', # Changed val_capsnet_acc
                                           save_best_only=True,
                                           save_weights_only=True,
                                           mode = 'max',
                                           verbose=1)
    earlystopper = callbacks.EarlyStopping(monitor='val_capsnet_acc',patience=100, verbose=0)
    #=========================ADDED BY Patrick FOR CSV FILE and others======================

    #save best True saves only if the metric improves
    #chk = ModelCheckpoint(MODEL_PATHN, monitor='val_capsnet_loss', save_best_only=False)
    #callbacks_list = [chk]
    #pass callback on fit
    #history = model.fit(X, Y, ... , callbacks=callbacks_list)



    # callbacks
    #log = callbacks.CSVLogger(save_dir + '/logRSNA(LSCaps).csv')          #save in models/log.csv
    log = callbacks.CSVLogger(save_dir + '/BrainCancer(CapsNet(MLAF-CapsNet)).csv')
    #tb = callbacks.TensorBoard(log_dir=save_dir + '/logs',    # models/logs/individual_log_files
    #                           batch_size=BATCH_SIZE, histogram_freq=int(10))# Take batch_size from hyperparams.py file


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    #========================================================================
    ## Define list of parameters
    #lrRate = [0.1, 0.01, 0.01, 0.0001, 0.002, 0.003]


    # Compile the training model
    start = datetime.today()  #**********Start tracking the training time****Line Added by Patrick
    ## Below fit is for Capsule Model
    '''
    train_model.compile(optimizer=optimizers.Adam(lr=LearningRate),
                        loss=[margin_loss, 'mse'],
                        #loss ='binary_crossentropy', # classification error & reconstruction error ###mse
                        loss_weights=[1., LAM_RECON],
                        metrics={'capsnet': 'accuracy'})
                        '''
    # This function keeps the initial learning rate for the first ten epochs
# and decreases it exponentially after that.
    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)

            #model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
            #model.compile(tf.keras.optimizers.SGD(), loss='mse')
            #round(model.optimizer.lr.numpy(), 5)
    # compile the model

    train_model.compile(optimizer=optimizers.Adam(lr=LearningRate),
                  loss=[margin_loss, pose_loss], ### original
                  #loss = [contrastive_loss, pose_loss],
                  #loss = [triplet_loss, pose_loss],
                  #loss_weights=[0,1],
                  loss_weights=[1.,1], ## orginal
                  metrics={'capsnet': 'accuracy'})

    #round(train_model.optimizer.lr.numpy(), 5)
    #callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    #learn_control = ReduceLROnPlateau(monitor='val_capsnet_acc', patience=5, verbose=1, factor=0.2)#min_lr=1e-7
    # Train model


    train_model.fit([x_train, y_train], ## Changed Train_model to model for the baselines
                    [y_train, x_train],
                    #steps_per_epoch = samples // BATCH_SIZE,
                    batch_size = BATCH_SIZE,
                    epochs=EPOCHS,
                    shuffle=True,
                    validation_data=[[x_test,y_test],
                                    [y_test, x_test]],
                    #validation_steps = val_sample // BATCH_SIZE,
                    #callbacks=[learn_control, checkpoint, log])
                    callbacks =[earlystopper, checkpoint, log])
    #round(train_model.optimizer.lr.numpy(), 5)



    train_model.load_weights(MODEL_PATH)

    eval_model.load_weights(MODEL_PATH)




   

    ## Evaluating the Model
    y_pred = eval_model.predict(x_test)

    
    from sklearn.metrics import classification_report, confusion_matrix
    #y_pred  = np.argmax(y_pred, axis=0)
    #y_test = np.argmax(y_test, axis=0)

    #num_samples = 3234
    y_pred = np.array(y_pred[0])
    #print("y_pred", y_pred)
    #round_pred = (y_pred.argmax(), axis=0)
    round_pred = y_pred.argmax(axis=1)
    #print("round_predict",round_pred.shape)

    y_test=np.argmax(y_test, axis=1)
    #print("y_test", y_test.shape)

    #print("y_predict",y_pred.shape)
    #round_pred=np.argmax(round_pred, axis=0)

    ########### Print and Plot Confusion Matrix ################
    cnf_matrix = confusion_matrix(y_test, round_pred)
    print(cnf_matrix)

    import pandas as pd
    import seaborn as sn


    

    



    #=========================ADDED BY ADU FOR CSV FILE and others======================
    end = datetime.today()  #**********Stop tracking the training time****Line Added by ADU


    print("*"*10)
    print("Training Time Took: {}".format(end - start))
    print("*"*10)




    from utils import plot_log
    
    plot_log(save_dir + '/BrainCancer(CapsNet(MLAF-CapsNet)).csv', show=True)
    train_model.save('BrainCancer(CapsNet(MLAF-CapsNet))_model.h5')
   #=========================ADDED BY ADU FOR CSV FILE and others======================



# In[ ]:
