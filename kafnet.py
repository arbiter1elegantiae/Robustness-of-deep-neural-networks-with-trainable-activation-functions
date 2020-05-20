
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


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



from tensorflow.keras.layers import Layer

class Kaf(Layer):
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
