
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
    def __init__(self, ratio=0.2, conv = False, data_format="channels_last", **kwargs):

        super(Kwta, self).__init__(**kwargs)

        self.conv = conv
        self.ratio = ratio
        self.data_format = data_format
        

    def build(self, input_shape):
        
        # dim = C*H*W when conv=True, #units otherwise
        self.dim = tf.math.reduce_prod(input_shape[1:]).numpy() 



    def call(self, inputs):
        
        # In case of incremental learning we want to update k 
        k = int(math.ceil(self.ratio*self.dim))

        # Store input shape to the layer
        shape = tf.shape(inputs)
        
        if self.conv:
            
            if self.data_format == "channels_last":
                # Transpose to have channel_first format
                inputs = tf.transpose(inputs, perm = [0, 3, 1, 2])

            # Flatten the tensor
            inputs = tf.reshape(inputs, [-1, self.dim])
        
        # Get the largest K-th value from each flattened tensor in the batch
        kths_largest = tf.math.top_k(inputs, k, sorted=True)[0][:,-1]
        
        # Set to 0 all the values but the K largests per batch
        outputs = tf.where(tf.math.less(inputs, tf.expand_dims(kths_largest, -1)), 0.0, inputs)

        if self.conv:
            # Transform back to the original shape
            outputs = tf.reshape(outputs, shape)
        
        return outputs

    
    def get_config(self):
        
        my_config = {
            
            'conv' : self.conv,
            'ratio' : self.ratio,
            'data_format' : self.data_format,
            'dim' : self.dim
        }

        base_config = super(Kwta, self).get_config()
        return dict(list(base_config.items()) + list(my_config.items()))

