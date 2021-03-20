#### MLAF-CapsNet

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
from layers import PrimaryCap, CapsuleLayer, CapsLength, Mask, LBC,gabor_init, custom_gabor, CAN, PrimaryCap2, Clahe, enhance
from Improved_CLAHE import improved_clahe
from keras.layers import Input,Conv2D, MaxPool2D, Dense, Flatten, BatchNormalization, Reshape, Lambda, Layer
from keras.layers.convolutional import AtrousConvolution2D
from keras.layers import concatenate, GlobalAvgPool2D, Dropout, LeakyReLU #, ReLU, Activation
from hyperparams import * # Check the hyper parameters in 'hyperparams.py'
import keras
import pydotplus
import keras.utils
from keras.layers import Activation
#from activationf.Rtanh import rtanh
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

from keras.layers import Input,Flatten, merge







#from IPython.display import SVG
#from keras.utils.vis_utils import model_to_dot
#keras.utils.vis_utils_pydot = pydotplus


def GetDiag(var):
    from theano.tensor.nnet.conv3d2d import DiagonalSubtensor
    takeDiag = DiagonalSubtensor()
    [s1,s2,s3,s4]=var.shape
    diag=takeDiag(var,2,3)
    a=diag.reshape((s1,s2,1,s3)).repeat(s3,axis=2)
    b=diag.reshape((s1,s2,s3,1)).repeat(s3,axis=3)
    return a+b

def out_diag_shape(input_shape):
    return input_shape

#### Model
def base_model(input_shape, output_shape):
    routings = 3


    

    input_x = Input(input_shape)
    _routings = ROUTING
    _digitcap_num = NUM_CLASSES
    _digitcap_dim = DIGIT_CAP_DIM
    _digitcap_dimm = DIGIT_CAP_DIM2

    nb_filters = 64
    


    ######################### ENCODER LAYER ###################################

    ## First Convolution Block, consist of 4 Convs and 1 PrimaryCaps with short skips
    #
    x = enhance()(input_x)
    #input_x = improved_clahe()(input_x)
    #print('clahe', input_x)
    conv1 = AtrousConvolution2D(32, 1, 1, atrous_rate=(2,2), border_mode='same', kernel_initializer='he_normal')(x)
    act0 = Activation('tanh')(conv1)
    bn0 = layers.BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None)(act0)
    mx0 = MaxPool2D(2, strides = 2, padding='same')(bn0)

    conv2 = AtrousConvolution2D(32, 1, 1, atrous_rate=(4,4), border_mode='same', kernel_initializer='he_normal' )(x)
    act1 = Activation('tanh')(conv2)
    bn1 = layers.BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None)(act1)
    mx1 = MaxPool2D(2, strides = 2, padding='same')(bn1)

    conv3= AtrousConvolution2D(32, 1, 1, atrous_rate=(6,6), border_mode='same', kernel_initializer='he_normal' )(x)
    act2 = Activation('tanh')(conv3)
    bn2 = layers.BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None)(act2)
    mx2 = MaxPool2D(2, strides = 2, padding='same')(bn2)


    con1 = layers.concatenate([mx0, mx1, mx2], axis=1)



    conv4 = Conv2D(64, 3, strides = 2, padding = 'valid', activation = 'tanh', kernel_initializer='he_normal', name='lane_1_Conv2')(con1)
    bn3 = layers.BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None)(conv4)
    mx3 = MaxPool2D(2, strides = 2, padding='same')(bn3)

    conv5 = Conv2D(64, 3, strides = 2, padding = 'valid', activation = 'tanh', kernel_initializer='he_normal', name='lane_2_Conv2' )(con1)
    bn4 = layers.BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None)(conv5)
    mx4 = MaxPool2D(2, strides = 2, padding='same')(bn4)

    conv6 = Conv2D(64, 3, strides = 2, padding = 'valid', activation = 'tanh', kernel_initializer='he_normal', name='lane_3_Conv2')(con1)
    bn5 = layers.BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None)(conv6)
    mx5 = MaxPool2D(2, strides = 2, padding='same')(bn5)

    #con2 = layers.concatenate([bn3, bn4], axis=1)
    #con3 = layers.concatenate([con2, bn5], axis=1)


    con2 = layers.concatenate([mx3, mx4], axis=1)
    con3 = layers.concatenate([con2, mx5], axis=1)

    

    conv7 = Conv2D(128, 1, strides = 1, padding = 'valid', activation = 'tanh', kernel_initializer='he_normal', name='lane_1_Conv3')(con2)

    conv8 = Conv2D(128, 1, strides = 1, padding = 'valid', activation = 'tanh', kernel_initializer='he_normal', name='lane_2_Conv3' )(con2)

    #conv9 = AtrousConvolution2D(128, 3, 3, atrous_rate=(1,1), border_mode='same')(con3)
    #act4 = Activation('relu')(conv9)
    #bn5 = layers.BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None)(act4)
    #mx5 = MaxPool2D(2, strides = 2, padding='same')(bn5)
    conv9 = Conv2D(128, 1, strides = 1, padding = 'valid', activation = 'tanh', kernel_initializer='he_normal', name='lane_3_Conv3')(con3)

    con4 = layers.concatenate([conv7, conv8], axis=1)
    #print('con4:', con4)

    #con5 = layers.concatenate([con4, mx5], axis=1)
    #print('con4', con5)

    ######################### DECODER LAYER ###################################
    conv10 = Conv2D(128, 1, strides = 1, padding = 'valid', activation = 'tanh', kernel_initializer='he_normal', name='Conv4')(con4)
    bn3 = layers.BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None)(conv10)
    con6 = layers.concatenate([conv9, bn3], axis=1)



    #primarycaps1 = PrimaryCap(conv10, dim_capsule=8, n_channels=8, kernel_size=3,strides=2,padding='valid')



    ## Define the Primary capsule (PrimaryCaps)
    primarycaps = PrimaryCap(con6, dim_capsule=8, n_channels=32, kernel_size=3,strides=2,padding='valid')
    #primarycaps = PrimaryCap(bn7, dim_capsule_attr=1, num_capsule=32, kernel_size=3, strides=2, padding='valid')
    #print('Primary_Caps',primarycaps)



    #con6 = layers.concatenate([primarycaps1, primarycaps], axis=1)

    #pc2 = PrimaryCap(output, dim_capsule=8, n_channels=16, kernel_size=7,strides=2,padding='valid')


    #output = concatenate([pc1, pc2])

    #attn_caps = CAN(num_capsule=NUM_CLASSES, dim_capsule_attr=1, routings=routings, num_instance=n_instance, num_part=n_part,
                    #name='digitcaps')(primarycaps)

    #print('attn_caps', attn_caps)

    x = CapsuleLayer(num_capsule = _digitcap_num,dim_capsule = _digitcap_dim,routings = _routings,name='digitcaps')(primarycaps)
    #x1 = CapsuleLayer(num_capsule = 16, dim_capsule = _digitcap_dim,routings = _routings,name='digitcaps')(primarycaps)

    #x = layers.concatenate([x0,x1], axis=1)
    #x = layers.concatenate([x, attn_caps])
    #print('concat_x',x)




    #####################################################

    digitcaps = x
    print("digitcaps", digitcaps.shape)


    x = CapsLength(name='capsnet')(digitcaps)
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
    decoder.add(layers.Dense(512, input_dim= _digitcap_dim * _digitcap_num, init='he_normal'))#(digitcaps) init='uniform'
    decoder.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))
    decoder.add(Activation('tanh'))
    decoder.add(layers.Dropout(0.4))
    #decoder.add(layers.Batch_Normalization())
    decoder.add(layers.Dense(1024, init='he_normal'))
    decoder.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))
    decoder.add(Activation('tanh'))
    decoder.add(layers.Dropout(0.4))
    #decoder.add(layers.BatchNormalization())
    decoder.add(layers.Dense(np.prod(input_shape), init='he_normal'))
    decoder.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))
    decoder.add(Activation('sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))


    #train_model = models.Model([input_x], [out_caps, decoder(out_pose)]) ## New line added
    #eval_model = models.Model([input_x], [out_caps,decoder(out_pose)])
    # Models for training and evaluation (prediction)
    train_model = models.Model([input_x, y_label], [y_pred, decoder(true_digitcap)])
    eval_model = models.Model(input_x, [y_pred, decoder(maxlen_digitcap)])
    pc_model=models.Model(input_x, primarycaps)
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

'''
def manipulate_latent(model, data, args):
    print('-'*30 + 'Begin: manipulate' + '-'*30)
    x_test, y_test = data
    index = np.argmax(y_test, 1) == args.digit
    number = np.random.randint(low=0, high=sum(index) - 1)
    x, y = x_test[index][number], y_test[index][number]
    x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
    noise = np.zeros([1, 10, 16])
    x_recons = []
    for dim in range(16):
        for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
            tmp = np.copy(noise)
            tmp[:,:,dim] = r
            x_recon = model.predict([x, y, tmp])
            x_recons.append(x_recon)

    x_recons = np.concatenate(x_recons)

    img = combine_images(x_recons, height=16)
    image = img*255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + '/manipulate_RAFDB-%d.png' % args.digit)
    print('manipulated result saved to %s/manipulate-%d.png' % (args.save_dir, args.digit))
    print('-' * 30 + 'End: manipulate' + '-' * 30)
'''

if __name__ == "__main__":
    #lrRate = [0.1, 0.01, 0.01, 0.0001, 0.002, 0.003]


    #for lr in lrRate:
        #print('\nTraining with -->{0}<-- activation function\n'.format(lrRate))
    model = base_model(input_shape=(48,48,1), output_shape=(NUM_CLASSES,))
