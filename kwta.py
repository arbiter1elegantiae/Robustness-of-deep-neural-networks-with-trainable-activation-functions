
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import math

from tensorflow.keras.layers import Layer

class Kwta(Layer):
    """ 
    Parameters
    ----------
   """
    def __init__(self, ratio=0.2, conv = False, data_format="channels_last"):

        super(Kwta, self).__init__()

        self.conv = conv
        self.ratio = ratio
        self.data_format = data_format
        

    def build(self, input_shape):

        if not self.conv:
            self.k = int(math.ceil(self.ratio*input_shape[1]))

        else:
            dim = tf.math.reduce_prod(input_shape[1:]).numpy()
            self.k = int(math.ceil(self.ratio*dim))            

        self.shape = input_shape


    def call(self, inputs):

        if self.conv:

            if self.data_format == "channels_last":
                # Transpose to have channel dimension as the first dimension
                outputs = tf.transpose(inputs, perm = [0, 3, 1, 2])
            else:
                outputs = inputs

            # Flatten the tensor
            dim = tf.math.reduce_prod(self.shape[1:]) # dim = C*H*W
            outputs = tf.reshape(outputs, [-1, dim])
            # Get the largest K-th value from each flattened tensor in the batch
            kths_largest = tf.math.top_k(outputs, self.k, sorted=True)[0][:,-1]
            
            # Set to 0 all the values but the K largests per batch
            outputs = tf.where(tf.math.less(outputs, tf.expand_dims(kths_largest, -1)), 0.0, outputs)

            # Transform back to the original shape
            outputs = tf.reshape(outputs, shape = tf.shape(inputs))
        else:
            outputs = tf.math.top_k(inputs, self.k, sorted=True)[0]
        
        return outputs