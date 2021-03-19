# original source code from https://github.com/XifengGuo/CapsNet-Keras

import keras.backend as K
import tensorflow as tf
from keras import initializers, layers
from scipy.stats import bernoulli
#from sklearn.externals import joblib
from keras import Input
import joblib
import numpy as np
import cv2
import numpy as np

#from .attention import AttentionModule


class Mask(layers.Layer):
    """
    Mask a Tensor with shape=[None, num_capsule, dim_vector]
    either by the capsule with max length or by an additional
    input mask.
    Except the max-length capsule (or specified capsule),
    all vectors are masked to zeros. Then flatten the masked Tensor.

    For example:
        ```
        x = keras.layers.Input(shape=[8, 3, 2])  # batch_size=8, each sample contains 3 capsules with dim_vector=2
        y = keras.layers.Input(shape=[8, 3])  # True labels. 8 samples, 3 classes, one-hot coding.
        out = Mask()(x)  # out.shape=[8, 6]
        # or
        out2 = Mask()([x, y])  # out2.shape=[8,6]. Masked with true labels y. Of course y can also be manipulated.
        ```
    """
    def call(self, inputs, **kwargs):
        if type(inputs) is list:  # true label is provided with shape = [None, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # if no true label, mask by the max length of capsules. Mainly used for prediction
            # compute lengths of capsules
            x = K.sqrt(K.sum(K.square(inputs), -1))
            # generate the mask which is a one-hot code.
            # mask.shape=[None, n_classes]=[None, num_capsule]
            mask = K.one_hot(indices=K.argmax(x, 1), num_classes=x.get_shape().as_list()[1])

        # inputs.shape=[None, num_capsule, dim_capsule]
        # mask.shape=[None, num_capsule]
        # masked.shape=[None, num_capsule * dim_capsule]
        masked = K.batch_flatten(inputs * K.expand_dims(mask, -1))
        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # no true label provided
            return tuple([None, input_shape[1] * input_shape[2]])

    def get_config(self):
        config = super(Mask, self).get_config()
        return config


############### CLAHE as Base Layer ##########################
############### ADDED MY ME ##################################


class Clahe(layers.Layer):
    def __init__(self):
        super(Clahe, self).__init__()

    def contrast_limited_ahe(self, img_arr, level = 256, blocks = 8, threshold = 10.0, **args):
        ### equalize the distribution of histogram to enhance contrast, using CLAHE
        ### @params img_arr : numpy.array uint8 type, 2-dim
        ### @params level : the level of gray scale
        ### @params window_size : the window used to calculate CDF mapping function
        ### @params threshold : clip histogram by exceeding the threshold times of the mean value
        ### @return arr : the equalized image array
        (m, n) = img_arr.shape
        block_m = int(m / blocks)
        block_n = int(n / blocks)

        # split small regions and calculate the CDF for each, save to a 2-dim list
        maps = []
        for i in range(blocks):
            row_maps = []
            for j in range(blocks):
                # block border
                si, ei = i * block_m, (i + 1) * block_m
                sj, ej = j * block_n, (j + 1) * block_n

                # block image array
                block_img_arr = img_arr[si : ei, sj : ej]

                # calculate histogram and cdf
                hists = self.calc_histogram_(block_img_arr)
                clip_hists = self.clip_histogram_(hists, threshold = threshold)     # clip histogram
                hists_cdf = self.calc_histogram_cdf_(clip_hists, block_m, block_n, level)

                # save
                row_maps.append(hists_cdf)
            maps.append(row_maps)

        # interpolate every pixel using four nearest mapping functions
        # pay attention to border case
        arr = img_arr.copy()
        for i in range(m):
            for j in range(n):
                r = int((i - block_m / 2) / block_m)      # the row index of the left-up mapping function
                c = int((j - block_n / 2) / block_n)      # the col index of the left-up mapping function

                x1 = (i - (r + 0.5) * block_m) / block_m  # the x-axis distance to the left-up mapping center
                y1 = (j - (c + 0.5) * block_n) / block_n  # the y-axis distance to the left-up mapping center

                lu = 0    # mapping value of the left up cdf
                lb = 0    # left bottom
                ru = 0    # right up
                rb = 0    # right bottom

                # four corners use the nearest mapping directly
                if r < 0 and c < 0:
                    arr[i][j] = maps[r + 1][c + 1][img_arr[i][j]]
                elif r < 0 and c >= blocks - 1:
                    arr[i][j] = maps[r + 1][c][img_arr[i][j]]
                elif r >= blocks - 1 and c < 0:
                    arr[i][j] = maps[r][c + 1][img_arr[i][j]]
                elif r >= blocks - 1 and c >= blocks - 1:
                    arr[i][j] = maps[r][c][img_arr[i][j]]
                # four border case using the nearest two mapping : linear interpolate
                elif r < 0 or r >= blocks - 1:
                    if r < 0:
                        r = 0
                    elif r > blocks - 1:
                        r = blocks - 1
                    left = maps[r][c][img_arr[i][j]]
                    right = maps[r][c + 1][img_arr[i][j]]
                    arr[i][j] = (1 - y1) * left + y1 * right
                elif c < 0 or c >= blocks - 1:
                    if c < 0:
                        c = 0
                    elif c > blocks - 1:
                        c = blocks - 1
                    up = maps[r][c][img_arr[i][j]]
                    bottom = maps[r + 1][c][img_arr[i][j]]
                    arr[i][j] = (1 - x1) * up + x1 * bottom
                # bilinear interpolate for inner pixels
                else:
                    lu = maps[r][c][img_arr[i][j]]
                    lb = maps[r + 1][c][img_arr[i][j]]
                    ru = maps[r][c + 1][img_arr[i][j]]
                    rb = maps[r + 1][c + 1][img_arr[i][j]]
                    arr[i][j] = (1 - y1) * ( (1 - x1) * lu + x1 * lb) + y1 * ( (1 - x1) * ru + x1 * rb)
        arr = arr.astype("uint8")
        return arr






class CapsLength(layers.Layer):
    """
    Compute the length of vectors.
    This is used to compute a Tensor that has the same shape with y_true in margin_loss.
    Using this layer as model's output can directly predict labels
    by using `y_pred = np.argmax(model.predict(x), 1)`

    inputs: shape=[None, num_vectors, dim_vector]
    output: shape=[None, num_vectors]
    """
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1) + K.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        return config


def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule.
    It drives the length of a large vector to near 1 and small vector to 0

    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors



    ##### ADDED BY MYSELF ###
    ########### Horizontal Squash (Compressed) #############





def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    """
    Apply Conv2D `n_channels` times and concatenate all capsules
    :param inputs: 4D tensor, shape=[None, width, height, channels]
    :param dim_capsule: the dim of the output vector of capsule
    :param n_channels: the number of types of capsules
    :return: output tensor, shape=[None, num_capsule, dim_capsule]
    """
    #output = layers.Conv2D(filters=dim_capsule*n_channels,kernel_size=kernel_size, strides=strides,padding=padding,
                           #name='primarycap_conv2d')(inputs)
    output = layers.Conv2D(filters=dim_capsule*n_channels,kernel_size=kernel_size, strides=strides,padding=padding)(inputs)

    #outputs = layers.Reshape(target_shape=[-1, dim_capsule],name='primarycap_reshape')(output)
    outputs = layers.Reshape(target_shape=[-1, dim_capsule])(output)
    print('vector', outputs)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)
    #return layers.Lambda(k_means_squash, name='primarycap_squash')(outputs)
    #return layers.Lambda(k_means_squash)(outputs)
    #return layers.Lambda(horiz_squash)(outputs)
    #return layers.Lambda(vert_squash)(outputs)



def PrimaryCap3(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    """
    Apply Conv2D `n_channels` times and concatenate all capsules
    :param inputs: 4D tensor, shape=[None, width, height, channels]
    :param dim_capsule: the dim of the output vector of capsule
    :param n_channels: the number of types of capsules
    :return: output tensor, shape=[None, num_capsule, dim_capsule]
    """
    #output = layers.Conv2D(filters=dim_capsule*n_channels,kernel_size=kernel_size, strides=strides,padding=padding,
                           #name='primarycap_conv2d')(inputs)
    output = layers.Conv2D(filters=dim_capsule*n_channels,kernel_size=kernel_size, strides=strides,padding=padding)(inputs)

    #outputs = layers.Reshape(target_shape=[-1, dim_capsule],name='primarycap_reshape')(output)
    outputs = layers.Reshape(target_shape=[-1, dim_capsule])(output)
    print('vector', outputs)
    return layers.Lambda(squash, name='primarycap_squash2')(outputs)
    #return layers.Lambda(k_means_squash, name='primarycap_squash')(outputs)
    #return layers.Lambda(k_means_squash)(outputs)
    #return layers.Lambda(horiz_squash)(outputs)
    #return layers.Lambda(vert_squash)(outputs)




class CapsuleLayer(layers.Layer):
    """
    The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the
    neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
    from scalar to vector. So its input shape = [None, input_num_capsule, input_dim_capsule] and output shape = \
    [None, num_capsule, dim_capsule]. For Dense Layer, input_dim_capsule = dim_capsule = 1.

    :param num_capsule: number of capsules in this layer
    :param dim_capsule: dimension of the output vectors of the capsules in this layer
    :param routings: number of iterations for the routing algorithm
    """
    def __init__(self, num_capsule, dim_capsule, routings=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Transform matrix
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.built = True

    def call(self, inputs, training=None):
        # inputs.shape=[None, input_num_capsule, input_dim_capsule]
        # inputs_expand.shape=[None, 1, input_num_capsule, input_dim_capsule]
        inputs_expand = K.expand_dims(inputs, 1)

        # Replicate num_capsule dimension to prepare being multiplied by W
        # inputs_tiled.shape=[None, num_capsule, input_num_capsule, input_dim_capsule]
        inputs_tiled = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])

        # Compute `inputs * W` by scanning inputs_tiled on dimension 0.
        # x.shape=[num_capsule, input_num_capsule, input_dim_capsule]
        # W.shape=[num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
        # Regard the first two dimensions as `batch` dimension,
        # then matmul: [input_dim_capsule] x [dim_capsule, input_dim_capsule]^T -> [dim_capsule].
        # inputs_hat.shape = [None, num_capsule, input_num_capsule, dim_capsule]
        inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems=inputs_tiled)

        # Begin: Routing algorithm ---------------------------------------------------------------------#
        # The prior for coupling coefficient, initialized as zeros.
        # b.shape = [None, self.num_capsule, self.input_num_capsule].
        b = tf.zeros(shape=[K.shape(inputs_hat)[0], self.num_capsule, self.input_num_capsule])
        print("bij", b)

        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):
            # c.shape=[batch_size, num_capsule, input_num_capsule]
            c = tf.nn.softmax(b, dim=1) #orginal line of code

            #replace the coupling coeffience softmax with sigmoid
            #### Change by ME### ##
            #c = tf.nn.sigmoid(b)
            print('Cij', c)


            # c.shape =  [batch_size, num_capsule, input_num_capsule]
            # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
            # The first two dimensions as `batch` dimension,
            # then matmal: [input_num_capsule] x [input_num_capsule, dim_capsule] -> [dim_capsule].
            # outputs.shape=[None, num_capsule, dim_capsule]
            outputs = squash(K.batch_dot(c, inputs_hat, [2, 2]))  # [None, 10, 16]

            if i < self.routings - 1:
                # outputs.shape =  [None, num_capsule, dim_capsule]
                # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
                # The first two dimensions as `batch` dimension,
                # then matmal: [dim_capsule] x [input_num_capsule, dim_capsule]^T -> [input_num_capsule].
                # b.shape=[batch_size, num_capsule, input_num_capsule]
                b += K.batch_dot(outputs, inputs_hat, [2, 3])
        # End: Routing algorithm -----------------------------------------------------------------------#

        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])

    def get_config(self):
        config = {
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule,
            'routings': self.routings
        }
        base_config = super(CapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))





############################## Attention Capsulue##############################################


def PrimaryCap2(inputs,num_capsule, dim_capsule_attr, kernel_size, strides, padding):
    """
    Apply Conv2D `n_channels` times and concatenate all capsules
    :param inputs: 4D tensor, shape=[None, width, height, channels]
    :param dim_capsule: the dim of the output vector of capsule (not including geometric pose)
    :param n_channels: the number of types of capsules
    :return: output.shape=[None, num_capsule, num_instance, dim_capsule]
    """
    # The pose will contain a probability, a geometric pose data (i.e. location) and attributes.

    dim_capsule=dim_capsule_attr+dim_geom+1

    def build_geom_pose(x):
        '''
        build a PrimaryCap output from the Conv2D layer
        :param x: attributes from input Conv2D
        :return:
        '''
        _,rows,cols,num_capsule,dim_x = x.shape
        dim_capsule=dim_x-2+dim_geom+1
        # create the probability part
        s_squared_norm = K.sum(K.square(x), -1, keepdims=True)
        probability = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())

        # create the xy location part
        bsz=tf.shape(x)[0]

        xcoord, ycoord = tf.meshgrid(tf.linspace(-1.0, 1.0, rows),
                                     tf.linspace(-1.0, 1.0, cols))
        xcoord = tf.reshape(xcoord, [1, rows,cols,1, 1])
        ycoord = tf.reshape(ycoord, [1, rows,cols,1, 1])

        xcoordtiled = tf.tile(xcoord, [bsz,1,1,num_capsule, 1])
        ycoordtiled = tf.tile(ycoord, [bsz,1,1,num_capsule, 1])

        # create the rotation + scale part (assume scale of 1)
        n=int(affine_filters/2)
        cosa0,sina0=tf.reduce_sum(x[...,:n],axis=-1,keep_dims=True),tf.reduce_sum(x[...,n:n*2],axis=-1,keep_dims=True)
        r = tf.sqrt(tf.add(tf.square(cosa0),tf.square(sina0)))
        r = r+K.epsilon()
        cosa=cosa0/r
        sina=sina0/r
        affine=tf.concat([cosa,sina,-sina,cosa],axis=-1)

        # now assemble the capsule output
        attrs=x[...,affine_filters:]
        o1=tf.concat([probability,xcoordtiled, ycoordtiled,affine,attrs],axis=-1)
        o2=tf.reshape(o1,[bsz,rows*cols,num_capsule,dim_capsule],name="primary_cap_build_pose_output_reshaping")
        out=tf.transpose(o2,[0,2,1,3])
        #out=tf.Print(out,[out[0,0,0,:]],message="primary cap output",summarize=100)
        return out


    output = layers.Conv2D(filters=num_capsule*(dim_capsule_attr+affine_filters), kernel_size=kernel_size, strides=strides, padding=padding,
                           name='primarycap_conv2d')(inputs)
    _ , rows, cols, channels = output.shape

    attroutputs = layers.Reshape(target_shape=[int(rows),int(cols),num_capsule,dim_capsule_attr+affine_filters], name='primarycap_attributes')(output)

    outputs=layers.Lambda(build_geom_pose, name='primarycap')(attroutputs)

    return outputs





dim_geom=6 # Number of dimensions used for the geometric pose
affine_filters=2 # filters to drive affine transformation

def squash_scale(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale






def Decoder(masked):
    ## Yumi refactored
    x_recon = layers.Dense(512, activation='relu',name="FC1")(masked)
    x_recon = layers.Dense(1024, activation='relu',name="FC2")(x_recon)
    x_recon = layers.Dense(2304, activation='sigmoid',name="FC3")(x_recon)
    ## mse to ensure that the reconstructed images are close to the original image.
    x_recon = layers.Reshape(target_shape=[28, 28, 1], name='out_recon')(x_recon)
    return(x_recon)





    def _part_to_whole_predictions(self, x):
        """
        Estimate the pose of the whole given the pose of the part.
        :param x: set of poses to transform
        """
        # inputs.shape=[ input_num_capsule, input_num_instance, input_dim_capsule]
        # output.shape=[num_instance*num_capsule, num_parts*input_num_capsule*input_num_instance,dim_capsule]
        # xt.shape = [ input_num_capsule, num_instance, input_num_instance, input_dim_capsule]
        # xpart.shape = [ num_instance, input_num_instance, num_capsule, num_part, dim_x,input_num_capsule]
        # gpose.shape = [ input_num_capsule, num_instance, input_num_instance, dim_geom+1]
        xt = K.tile(K.expand_dims(x,1),[1,self.num_instance,1,1])

        tmp = K.reshape( xt[:,:,:,:1],[self.input_num_capsule,self.num_instance,self.input_num_instance,1,1,1])
        tmp = K.tile(tmp,[1,1,1,self.num_capsule,self.num_part,1])
        ppart=K.permute_dimensions(tmp,[1,2,3,4,5,0])

        gpose = K.concatenate([xt[:,:,:,1:dim_geom+1],K.ones_like(xt[:,:,:,:1])]) # add 1 col to allow x-y translate
        gpart = K.concatenate([K.expand_dims(K.dot(gpose[i],self.W1[i]),-1) for i in range(self.input_num_capsule)])
        apart = K.concatenate([K.expand_dims(K.dot(xt[i,:,:,dim_geom+1:],self.W2[i]),-1) for i in range(self.input_num_capsule)])
        whole=K.concatenate([ppart,gpart,apart],4)
        output=K.permute_dimensions(whole,[0,2,3,5,1,4])
        output=K.reshape(output,[self.num_instance*self.num_capsule,
                                 self.num_part*self.input_num_capsule*self.input_num_instance,self.dim_capsule])
        # output = tf.Print(output, [tf.shape(x)], message='x', summarize=16)
        # output = tf.Print(output, [x[0,18,1:3]], message='x ', summarize=3)
        # output = tf.Print(output, [gpose[0,0,0,:]], message='x gpose ', summarize=5)
        # output = tf.Print(output, [gpose[0,1,0,:]], message='x gpose ', summarize=5)
        # output = tf.Print(output, [gpart[0,0,0,0,0,:]], message='x gpart ', summarize=5)
        # output = tf.Print(output, [gpart[0,1,0,0,0,:]], message='x gpart ', summarize=5)
        return output

    def _best_guess(self, c, inputs_hat):
        '''
        Combine the predicted poses 'input_hats' weighted by c to come up with best_guess of the capsule poses
        :param c: weights to apply to the input poses
        :param inputs_hat: input poses
        :return: best guess at pose
        '''
        # c.shape=[None, num_capsule * num_instance, num_part * input_num_capsule * input_num_instance]
        # inputs_hat.shape = [None,num_instance * num_capsule, num_parts, input_num_capsule * input_num_instance, dim_capsule]
        # guess.shape = [None,num_instance * num_capsule,dim_capsule]

        # take the mean probility
        probability = tf.reduce_mean(inputs_hat[:,:,:,0:1],axis=2)

        # find the mean weighted geometric pose
        sum_weighted_geoms = K.batch_dot(c,inputs_hat[:,:,:,1:dim_geom+1], [2, 2])
        one_over_weight_sums = tf.tile(tf.expand_dims(tf.reciprocal(K.sum(c,axis=-1)),-1),[1,1,dim_geom])
        mean_geom =  one_over_weight_sums*sum_weighted_geoms

        # squash the weighted sum of attributes
        weighted_attrs = K.batch_dot(c,inputs_hat[:,:,:,dim_geom+1:], [2, 2])
        scale = squash_scale(weighted_attrs)

        # use the magnitude of the squashedweighted sum of attributes for probability
        probability = scale

        guess = layers.concatenate([probability,mean_geom,weighted_attrs])
        return guess

    def _agreement(self, outputs, inputs_hat):
        '''
        Measure the fit of each predicted poses to the best guess pose and return an adjustment value for the routing
        coefficients
        :param outputs: the best guess estimate of whole pose
        :param inputs_hat: the per part estimate of the whole pose
        :return: adjustment factor to the routing coefficients
        '''

        # outputs.shape = [None, num_instance * num_capsule, dim_capsule]
        # inputs_hat.shape = [None,num_instance * num_capsule, num_parts * input_num_capsule * input_num_instance, dim_capsule]
        # x_agree.shape = [None,num_instance * num_capsule, num_parts*input_num_capsule * input_num_instance],
        # b.shape=[None,num_instance * num_capsule, num_parts*input_num_capsule * input_num_instance]

        geom_agree = K.batch_dot(outputs[:,:,1:dim_geom+1], inputs_hat[:,:,:,1:dim_geom+1], [2, 3])
        attr_agree = K.batch_dot(outputs[:,:,dim_geom+1:], inputs_hat[:,:,:,dim_geom+1:], [2, 3])
        attr_agree *= 0.01

        # geom_agree=tf.Print(geom_agree, [outputs[0,0,:dim_geom+1]], message='agree guess ', summarize=5)
        # geom_agree=tf.Print(geom_agree, [inputs_hat[0,0,0,:dim_geom+1]], message='agree uhat ', summarize=5)
        # geom_agree=tf.Print(geom_agree, [geom_agree[0,0,0]], message='geom_agree ', summarize=5)
        # geom_agree=tf.Print(geom_agree, [attr_agree[0,0,0]], message='attr_agree ', summarize=5)
        # geom_agree=tf.Print(geom_agree, [tf.reduce_max(geom_agree),tf.reduce_min(geom_agree)], message='geom_agree max/min', summarize=5)
        # geom_agree=tf.Print(geom_agree, [tf.reduce_max(attr_agree),tf.reduce_min(attr_agree)], message='attr_agree max/min', summarize=5)

        return geom_agree+attr_agree

    def call(self, inputs, training=None):
        # inputs.shape=[None, input_num_capsule, input_num_instance, input_dim_capsule]
        # inputs_hat.shape=[None,num_instance*num_capsule,num_parts*input_num_capsule*input_num_instance,dim_capsule]

        inputs_hat = K.map_fn(lambda x: self._part_to_whole_predictions(x), elems=inputs)

        # Begin: Routing algorithm ---------------------------------------------------------------------#
        # The prior for coupling coefficient, initialized as zeros.
        # b.shape = [None, self.num_capsule, self.num_parts, self.input_num_capsule].
        b = K.tf.zeros(shape=[K.shape(inputs_hat)[0], self.num_instance*self.num_capsule,
                                 self.num_part*self.input_num_capsule*self.input_num_instance])

        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):
            # c.shape=[batch_size, num_instance*num_capsule, input_num_capsule]
            tmpb = K.reshape(b, [-1,self.num_capsule * self.num_instance*self.num_part,
                                        self.input_num_capsule * self.input_num_instance])

            # softmax for all outputs of each input_capsule*input_instance
            tmpc = K.tf.nn.softmax(tmpb, dim=1)
            c=K.reshape(tmpc,[-1,self.num_capsule * self.num_instance,
                                self.num_part*self.input_num_capsule * self.input_num_instance])

            #outputs.shape=[None,num_instance * num_capsule,dim_capsule]
            outputs = self._best_guess(c, inputs_hat)

            if i < self.routings - 1: #
                b += self._agreement(outputs, inputs_hat)

        # End: Routing algorithm -----------------------------------------------------------------------#
        outputs=K.reshape(outputs,[-1,self.num_instance,self.num_capsule,self.dim_capsule])
        outputs=K.permute_dimensions(outputs,[0,2,1,3])
        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.num_instance, self.dim_capsule])


