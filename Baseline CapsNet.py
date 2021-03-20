#### Baseline CapsNet

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:
import tensorflow as tf
from tensorflow import keras


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

from sklearn.metrics import confusion_matrix
import itertools

# original source code from https://github.com/XifengGuo/CapsNet-Keras
import numpy as np
#import tensorflow as tf
from keras import layers, models
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from keras import backend as K
from layers import PrimaryCap, CapsuleLayer, CapsLength, Mask, LBC,gabor_init, custom_gabor
from keras.layers import Input,Conv2D, MaxPool2D, Dense, Flatten, BatchNormalization, Reshape, Lambda, Layer
from keras.layers import concatenate, GlobalAvgPool2D, Dropout, LeakyReLU #, ReLU, Activation
from hyperparams import * # Check the hyper parameters in 'hyperparams.py'
import keras
import pydotplus
import keras.utils
from keras.layers import Activation
#from activationf.Rtanh import rtanh
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

from Pneumonia_models import Pro_Activations
#import ntanh_a, ntanh0, ntanh1, ntanh2, 'ntanh3, ntanh4

#from IPython.display import SVG
#from keras.utils.vis_utils import model_to_dot
#keras.utils.vis_utils_pydot = pydotplus


#### Model
def base_model(input_shape, output_shape):
    routings = 3


    def fire(x, fs, fe):
        s = Conv2D(fs, 1, activation = 'relu')(x)
        e1 = Conv2D(fe, 1, activation = 'relu')(s)
        e3  = Conv2D(fe, 3, padding = 'same', activation = 'relu', )(s)

        output = concatenate([e1, e3])
        return output

    input_x = Input(input_shape)
    _routings = ROUTING
    _digitcap_num = NUM_CLASSES
    _digitcap_dim = DIGIT_CAP_DIM

    conv1 = Conv2D(256, 9, strides = 1, padding = 'valid',   activation = 'relu',  kernel_initializer='he_normal', name='Conv1')(input_x)
    


    pc2 = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9,strides=2,padding='valid')



    x = CapsuleLayer(num_capsule = _digitcap_num,dim_capsule = _digitcap_dim,routings = _routings,name='digitcaps')(pc2)
    

    digitcaps = x
    print("digitcaps", digitcaps.shape)


    x = CapsLength(name='capsnet')(x)
    y_pred = x
    print('y_pred', y_pred)
    #y_pred=svmclf.predict(digitcaps)

    # For 'reconstruction' and also as a 'regulerizer',
    # we get y label as an input as well
    y_label = Input(shape=output_shape)
    print("y_label", y_label.shape)

    true_digitcap = Mask()([digitcaps, y_label])

    maxlen_digitcap = Mask()(digitcaps)



     # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, input_dim= _digitcap_dim * _digitcap_num, init='uniform'))#(digitcaps)
    decoder.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))
    decoder.add(Activation('relu'))
    decoder.add(layers.Dropout(0.4))
    #decoder.add(layers.Batch_Normalization())
    decoder.add(layers.Dense(1024, init='uniform'))
    decoder.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))
    decoder.add(Activation('relu'))
    decoder.add(layers.Dropout(0.4))
    #decoder.add(layers.BatchNormalization())
    decoder.add(layers.Dense(np.prod(input_shape), init='uniform'))
    decoder.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))
    decoder.add(Activation('sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([input_x, y_label], [y_pred, decoder(true_digitcap)])
    eval_model = models.Model(input_x, [y_pred, decoder(maxlen_digitcap)])
    pc_model=models.Model(input_x, pc2)
    digitcaps_model = models.Model(input_x, digitcaps)

    # summary
    train_model.summary()

    '''
    noise = Input(shape=(output_shape, 16))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    #output = Conv2D(n_classes, 1)(output)
    '''
    #output = GlobalAvgPool2D()(output)
    #output = Activation('softmax')(output)

    #model = Model(input, output)
    return train_model, eval_model, digitcaps_model, decoder



if __name__ == "__main__":
    #lrRate = [0.1, 0.01, 0.01, 0.0001, 0.002, 0.003]


    #for lr in lrRate:
        #print('\nTraining with -->{0}<-- activation function\n'.format(lrRate))
    model = base_model(input_shape=(48,48,1), output_shape=(NUM_CLASSES,))
