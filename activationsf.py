# A TensorFlow 2 implementation of the following activation functions 
    
    # K-Winners Take All as described in https://arxiv.org/pdf/1905.10510.pdf
    # Kernel-Based Activation Functions as described in https://arxiv.org/pdf/1707.04035.pdf

# Author: Federico Peconi https://github.com/arbiter1elegantiae

import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import callbacks


# ------ Kwta ------- #

class Kwta(layers.Layer):
    """
    Keep the K-largests neurons of a (N, 1) tensor, set to 0 the remaining ones.

    Parameters
    ----------
    ratio: float32 
       Denotes the proportion of neurons of the layer which are going to be kept by the Kwta
       K is computed layer-wise from this value

    conv: bool
        Indicates if the activations are coming from a convolutive layer or not
        in particular, there are two supported activations only:
            - A batch of flattened neurons of shape = (b, x)
            - A batch of 2DConvolutions of shape = (b, x, y, f)
        if the shape does not match with any of the latter, an error is thrown 

    data_format: string
        Either "channel_last" or "channel_first" admitted. Former is default
        To be used in accordance with the data_format of the layer on which we are
        calling Kwta

    References
    ----------
    Chang, X. and Peilin, Z. and Changxi, Z., 2020
    Enhancing Adversarial Defense by k-Winners-Take-All,
    International Conference on Learning Representations
    
    """
    def __init__(self, ratio=0.2, conv = False, data_format="channels_last", **kwargs):

        super(Kwta, self).__init__(**kwargs)

        self.conv = conv
        self.ratio = tf.Variable(ratio, trainable=False)
        self.data_format = data_format
        

    def build(self, input_shape):

        # Raise an exception if the input rank is not white listed
        try:
            input_shape.assert_has_rank(2)
        except ValueError:
            try:
                input_shape.assert_has_rank(4)
            except ValueError:
                raise ValueError("The input shape to Kwta must be either a dense batch (b, x) \n or a gridlike batch (b, x, y, f)")
        
        # dim = C*H*W when conv=True, #units otherwise
        self.dim = tf.math.reduce_prod(input_shape[1:]).numpy()
        super(Kwta, self).build(input_shape)


    def call(self, inputs):
        
        # In case of incremental learning we want to update k 
        k = tf.cast(tf.math.ceil(self.ratio*self.dim), dtype=tf.int32)

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



# Round with decimal precision in tf
def my_tf_round(x, decimals = 0):
    multiplier = tf.constant(10**decimals, dtype=x.dtype)
    return tf.round(x * multiplier) / multiplier



class incremental_learning_withDecreasing_ratio(callbacks.Callback):
    """
    Callback to icrementally adjust during training the Kwta ratio  every 2 epochs until desired ratio is reached.
    
    Parameters
    ----------
    delta: float32
        Decreasing value: ratio_n+1 = ratio_n - delta
        Default = 0.01

    end_ratio: float32
        Desired valued to reach with fine tuning
        
    num_layers: int
        Number of Kwta Layers in the model
        Warning: every Kwta Layer needs to be named 'kwta_i' where i the i-th kwta layer 
    
    """

    def __init__(self, delta = 0.01, end_ratio = 0.15, num_layers):
        super(incremental_learning_withDecreasing_ratio, self).__init__()
        self.delta = delta
        self.end_ratio = end_ratio
        self.num_layers = num_layers

    def on_epoch_begin(self, epoch, logs=None):
        # The update occurs at the beginning of every 2 epochs
        if epoch % 2 == 0: 
            for i in range(1, self.num_layers + 1): # For each kwta layer
                name = 'kwta_'+str(i)
                layer = self.model.get_layer(name = name)
                layer.ratio.assign_sub(self.delta)
            
            print('\n Fine tuning: current ratio {:.2f} \n'.format(layer.ratio.numpy()))
    
    def on_epoch_end(self, epoch, logs=None):
        layer = self.model.get_layer('kwta_1')
        if ( my_tf_round(layer.ratio, 2) == self.end_ratio ) and epoch % 2 == 1: 
                print('\n Desired Ratio reached, stop training...')
                self.model.stop_training = True




# ------ Kaf ------- #

class Kaf(layers.Layer):
    """ 
    Kernel Activation Function implemented as a keras layer to allow parameters learning
    Detailed informations about the activation function can be found in the referenced paper 
    
    Parameters
    ----------
    D: int 
       dictionary size

    conv: bool
        indicates if the activations are coming from a convolutive layer or not
        in particular, there are two supported activations only:
            - A batch of flattened units i.e. of shape = (b, x)
            - A batch of 2DConvolutions i.e. of shape = (b, x, y, f) where f is supposed to be the channels size
        if the shape does not match with any of the latter, an error is thrown 

    ridge: string
        \in {tanh, elu, None} 
        specifies how the mixing coefficients need to be initialized in order to approximate the resulting 
        to either a tanh or elu activation function. If None than mixing coefficients are initialized randomly
        again, if ridge assumes other values, a ValueError is fired

    References
    ----------
    [1] Scardapane, S., Van Vaerenbergh, S., Totaro, S. and Uncini, A., 2019. 
        Kafnets: kernel-based non-parametric activation functions for neural networks. 
        Neural Networks, 110, pp. 19-32.
   """
    def __init__(self, D, conv=False, ridge=None, **kwargs):

        super(Kaf, self).__init__()
        
        # Init constants
        self.D = D
        self.conv = conv
        self.ridge = ridge
        
        step, dict = dictionaryGen(D)
        self.d = tf.stack(dict)
        self.k_bandw = 1/(6*(np.power(step,2)))
        

    def build(self, input_shape):

        # Raise an exception if the input rank is not white listed
        try:
            input_shape.assert_has_rank(2)
        except ValueError:
            try:
                input_shape.assert_has_rank(4)
            except ValueError:
                raise ValueError("The input shape for Kaf must be either a dense batch (b, x) \n or a gridlike batch (b, x, y, f)")

        # Init mix coefficients
        if self.ridge is not None:
            
            eps = 1e-06 # stick to the paper's design choice
            
            if self.ridge == 'tanh':
                t = tf.keras.activations.tanh(self.d)
                

            elif self.ridge == 'elu':
                t = tf.keras.activations.elu(self.d)

            else:
                raise ValueError("The Kaf layer supports approximation only for 'tanh' and 'elu'")

            K = kernelMatrix(self.d, self.k_bandw)
            
            a = tf.reshape(np.linalg.solve(K + eps*tf.eye(self.D), t), shape=(1, 1, -1)) # solve ridge regression and get 'a' coeffs
            a_init = a * tf.ones(shape=(1, input_shape[-1], self.D)) # reshape a
           
            self.a = tf.Variable(initial_value=a_init, trainable=True, name = 'mix_coeffs')

        else:
            # regularizer_l2 = tf.keras.regularizers.l2(0.0001)
            self.a = self.add_weight(shape=(1, input_shape[-1], self.D),
                                 name = 'mix_coeffs',   
                                 initializer= 'random_normal',
                                 #regularizer= regularizer_l2,
                                 trainable=True) 
        
        # Adjust dimensions in order to exploit broadcasting and compute the entire batch all at once
        if not self.conv:
            self.d = tf.Variable(tf.reshape(self.d, shape=(1, 1, self.D)), name='dictionary', trainable=False)
        
        else:
            self.a = tf.Variable(tf.reshape(self.a, shape=(1,1,1,-1,self.D)), name = 'mix_coeffs')
            self.d = tf.Variable(tf.reshape(self.d, shape=(1, 1, 1, 1, self.D)), name='dictionary', trainable=False)


    def call(self, inputs):
        inputs = tf.expand_dims(inputs, -1)
        return kafActivation(inputs, self.a, self.d, self.k_bandw)



def dictionaryGen(D):
    """ 
    Dictionary generator

    Parameters
    ----------
    D: int 
       dictionary size

    Returns
    -------
    tuple    
    (the step size \gamma, np array of D integers evenly distributed around 0)
         
    """
    d_pos = np.linspace(-2, 2, num= D, retstep=True, dtype=np.float32)
    return (d_pos[1], d_pos[0])



def kafActivation(x, a, d, k_bwidth):
    """
    For each element in x, compute the weighted sum of the 1D-Gaussian kernel
    
    Parameters
    ----------
    x: tensor tf.float32
       each element of the tensor is an activation for the kaf
    
    a: tensor tf.float32
       tensor of mixing coefficients
       
    d: tensor tf.int32
       dictionary tensor
    
    k_bwidth: tf.float32
            kernel bandwidth
    """
    x = tf.math.square(x - d)
    x = a * tf.math.exp((-k_bwidth) * x)

    return tf.reduce_sum(x, -1)



def kernelMatrix(d, k_bwidth):
    """ 
    Return the kernel matrix K \in R^D*D where K_ij = ker(d_i, d_j) 
    """
    d = tf.expand_dims(d, -1)
    return tf.exp(- k_bwidth * tf.square(d - tf.transpose(d)))
