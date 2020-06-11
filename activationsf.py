# A TensorFlow2 implementation of the following activation functions 
    # K-Winners Take All as described in https://arxiv.org/pdf/1905.10510.pdf
    # Kernel-Based Activation Functions as described in https://arxiv.org/pdf/1707.04035.pdf

# Author: Federico Peconi https://github.com/arbiter1elegantiae

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow import math 

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
        k = tf.cast(math.ceil(self.ratio*self.dim), dtype=tf.int32)

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
        
        config = super(Kwta, self).get_config()
        config.update({
            'conv': self.conv,
            'data_format': self.data_format
            })
        return config




def my_tf_round(x, decimals = 0):
    """
    Round with decimal precision in tf
    """
    multiplier = tf.constant(10**decimals, dtype=x.dtype)
    return tf.round(x * multiplier) / multiplier



class incremental_learning_withDecreasing_ratio(callbacks.Callback):
    """
    Callback to icrementally adjust during training the Kwta ratio every 2 epochs until desired ratio is reached.
    
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

    def __init__(self, num_layers, delta = 0.01, end_ratio = 0.15):

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

        super(Kaf, self).__init__(**kwargs)
        
        # Init constants
        self.D = D
        self.conv = conv
        self.ridge = ridge
        
        step, dict = dictionaryGen(D)
        self.d = tf.cast( tf.stack(dict), dtype=tf.float16 )
        self.k_bandw = tf.cast( 1/(6*(tf.math.pow(step,2))), dtype=tf.float16)
        

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
        regularizer_l2 = tf.keras.regularizers.l2(0.0005)
        if self.conv:
          self.a = self.add_weight(shape=(1, 1, 1, input_shape[-1], self.D),
                                name = 'mix_coeffs',   
                                initializer= 'random_normal',
                                regularizer= regularizer_l2,
                                trainable=True) 
        else:
          self.a = self.add_weight(shape=(1, input_shape[-1], self.D),
                                 name = 'mix_coeffs',   
                                 initializer= 'random_normal',
                                 regularizer= regularizer_l2,
                                 trainable=True) 

        if self.ridge is not None:
            eps = 1e-06 # stick to the paper's design choice
            
            if self.ridge == 'tanh':
                t = tf.keras.activations.tanh(self.d)
                
            elif self.ridge == 'elu':
                t = tf.keras.activations.elu(self.d)

            else:
                raise ValueError("The Kaf layer supports approximation only for 'tanh' and 'elu'")

            K = kernelMatrix(self.d, self.k_bandw)
            

            # Compute and adjust mix coeffs 
            if self.conv:
              a = tf.reshape(np.linalg.solve(tf.cast(K, dtype=tf.float32) + eps*tf.eye(self.D), tf.cast(t, dtype=tf.float32)), shape=(1, 1, 1, 1, -1)) # solve ridge regression and get 'a' coeffs
              a = a * tf.ones(shape=(1, 1, 1, input_shape[-1], self.D))
              
            else:
              a = tf.reshape(np.linalg.solve(tf.cast(K, dtype=tf.float32) + eps*tf.eye(self.D), tf.cast(t, dtype=tf.float32)), shape=(1, 1, -1)) # solve ridge regression and get 'a' coeffs
              a = a * tf.ones(shape=(1, input_shape[-1], self.D))
            
            # Set mix coeff
            # Weights are stored in a list, thus we need to add a first dimension as the list index
            self.set_weights(tf.expand_dims(a, 0))

        # Adjust dictionary dimension
        if not self.conv:
            self.d = tf.Variable(tf.reshape(self.d, shape=(1, 1, self.D)), name='dictionary', trainable=False)
        
        else:
            self.d = tf.Variable(tf.reshape(self.d, shape=(1, 1, 1, 1, self.D)), name='dictionary', trainable=False)

 
    def call(self, inputs):
        
        inputs = tf.expand_dims(inputs, -1)
        return kafActivation(inputs, self.a, self.d, self.k_bandw)
    

    def get_config(self):
        
        config = super(Kaf, self).get_config()
        config.update({
            'D': self.D,
            'conv': self.conv,
            'ridge': self.ridge
            })
        return config



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
    x = math.multiply(a, math.exp(math.multiply(-k_bwidth, math.square(x - d))))
    return tf.reduce_sum(x, -1)



def kernelMatrix(d, k_bwidth):
    """ 
    Return the kernel matrix K \in R^D*D where K_ij = ker(d_i, d_j) 
    """
    d = tf.expand_dims(d, -1)
    return math.exp(- k_bwidth * math.square(d - tf.transpose(d)))



class plot_kafs_epoch_wise(callbacks.Callback):
    '''
    Plot learned Kafs during training for every epoch. One Kaf is displayed per layer

    Parameters
    ----------
    num_layers: integer
                number of Kaf Layers in the model
    
    Warning: every Kaf Layer needs to be named 'kaf_i' where i the i-th kaf layer    
    '''
    def __init__(self, num_layers):
        
        super(plot_kafs_epoch_wise, self).__init__()
        self.num_layers = num_layers

    
    def on_epoch_begin(self, epoch, logs=None):
        
        # Get Kaf's invariants: kernel bandwidth and dictionary
        kaf1 = self.model.get_layer(name = 'kaf_1')
        kb = kaf1.k_bandw
        d_tmp = tf.squeeze(kaf1.d)
        d = tf.expand_dims(d_tmp, 0)
        
        # We want to evaluate Kafs on the same input: use dictionary itself as activation
        x = tf.expand_dims(d_tmp, -1)

        # Prepare plot settings
        fig=plt.figure(figsize=(15, 8))
        plt.subplots_adjust(wspace = 0.5, hspace = 0.3)
        columns = int(self.num_layers/2) + 1
        rows = 2
        ax = []
        
        for i in range(1, self.num_layers + 1): # For each Kaf layer
          
          name = 'kaf_'+str(i)
          layer = self.model.get_layer(name = name)
          
          # Get mixing coefficients and compute Kaf
          a = tf.cast( tf.expand_dims(tf.squeeze(layer.a)[0], 0), dtype = tf.float16 )
          kaf = kafActivation(x, a, d, kb)

          # Plot
          ax.append( fig.add_subplot(rows, columns, i) )
          ax[-1].set_title('{}, Epoch {}'.format(name,epoch+1))  
          plt.plot(d_tmp, kaf, 'r')
        
        plt.show()

       

            
